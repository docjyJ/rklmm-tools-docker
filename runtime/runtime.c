#include <stdbool.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <argp.h>
#include "rkllm.h"

#define MAX_INPUT_LENGTH 1024
#define PROMPT_USER "User: "
#define PROMPT_ASSISTANT "Assistant: "
#define PROMPT_FORMATTED "User: %sAssistant: "


static char doc[] = "Argp example #3 -- a program with options and arguments using argp";

static char args_doc[] = "MODEL_FILE";

static struct argp_option options[] = {
        {"num_npu", 'r', "NUM_NPU", 0, "Number of NPUs to use"},
        {0}
};

struct arguments {
    char *model_file;
    int32_t num_npu;
};


static error_t parse_opt(int key, char *arg, struct argp_state *state) {
    struct arguments *arguments = state->input;

    switch (key) {
        case 'r':
            arguments->num_npu = arg ? atoi(arg) : 2;
            break;
        case ARGP_KEY_ARG:
            if (state->arg_num >= 1)
                argp_usage(state);
            arguments->model_file = arg;
            break;
        case ARGP_KEY_END:
            if (state->arg_num < 1)
                argp_usage(state);
            break;
        default:
            return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

/* Our argp parser. */
static struct argp argp = {options, parse_opt, args_doc, doc};


LLMHandle llmHandle = NULL;

void exit_handler(int signal) {
    if (llmHandle != NULL)
        rkllm_destroy(llmHandle);

    exit(signal);
}


char user_buffer[MAX_INPUT_LENGTH];

char assistant_buffer[MAX_INPUT_LENGTH + sizeof PROMPT_FORMATTED - 2];

void callback(RKLLMResult *result, void *userdata, LLMCallState state) {
    if (state == LLM_RUN_FINISH)
        printf("\n");
    else if (state == LLM_RUN_ERROR)
        exit_handler(1);
    else
        printf("%s", result->text);
}


int main(int argc, char **argv) {
    signal(SIGINT, exit_handler);

    struct arguments arguments = {NULL, .num_npu = 2};

    argp_parse(&argp, argc, argv, 0, 0, &arguments);

    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path = arguments.model_file;
    param.num_npu_core = arguments.num_npu;
    param.top_k = 1;
    param.max_new_tokens = 256;
    param.max_context_len = 512;
    param.logprobs = false;
    param.top_logprobs = 5;
    param.use_gpu = false;

    rkllm_init(&llmHandle, param, callback);


    while (true) {
        printf(PROMPT_USER);

        if (fgets(user_buffer, MAX_INPUT_LENGTH, stdin) == NULL)
            exit_handler(0);

        size_t len = strlen(user_buffer);
        if (len > 0 && user_buffer[len - 1] == '\n')
            user_buffer[len - 1] = '\0';

        sprintf(assistant_buffer, PROMPT_FORMATTED, user_buffer);

        printf(PROMPT_ASSISTANT);

        rkllm_run(llmHandle, assistant_buffer, NULL);
    }
}

