python -u -m vllm.entrypoints.openai.api_server \
       --model TheBloke/Mistral-7B-Instruct-v0.2-GPTQ \
       --port 3000 \
       --dtype half \
       --quantization gptq
