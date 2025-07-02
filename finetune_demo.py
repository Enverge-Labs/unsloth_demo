import marimo

__generated_with = "0.14.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""# Welcome to Enverge Lab""")
    return


@app.cell
def _():
    from unsloth import FastLanguageModel
    import torch
    return (FastLanguageModel,)


@app.cell
def _():
    max_seq_length = 2048
    dtype = None
    load_in_4bit = True
    return dtype, load_in_4bit, max_seq_length


@app.cell
def _(FastLanguageModel, dtype, load_in_4bit, max_seq_length):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit
    )
    return model, tokenizer


@app.cell
def _(FastLanguageModel, model):
    model_peft = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None
    )
    return (model_peft,)


@app.cell
def _(mo):
    mo.md(r"""Note: PyTorch contains some helper classes for these formats: [read more](https://docs.pytorch.org/torchtune/0.2/generated/torchtune.data.AlpacaInstructTemplate.html).
    
    Not used in this demo to emphasize what's happening under the hood.
    """)
    return


@app.cell
def _(tokenizer):
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. Finally, an explanation justifies the generated.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    SQL: {}

    Explanation: {}
    """

    EOS_TOKEN = tokenizer.eos_token


    def formatting_prompts_func(examples):
        company_databases = examples["sql_context"]
        prompts = examples["sql_prompt"]
        sqls = examples["sql"]
        explanations = examples["sql_explanation"]
        texts = []
        for company_database, prompt, sql, explanation in zip(company_databases, prompts, sqls, explanations):
            text = alpaca_prompt.format(prompt, company_database, sql, explanation) + EOS_TOKEN
            texts.append(text)
        return {"text": texts}
    return (formatting_prompts_func,)


@app.cell
def _():
    from datasets import load_dataset
    return (load_dataset,)


@app.cell
def _(formatting_prompts_func, load_dataset):
    dataset = load_dataset("gretelai/synthetic_text_to_sql", split="train")
    dataset = dataset.map(formatting_prompts_func, batched=True)
    return (dataset,)


@app.cell
def _(dataset, max_seq_length, model_peft, tokenizer):
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from unsloth import is_bfloat16_supported

    trainer = SFTTrainer(
        model = model_peft,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_train_epochs = 1,
            max_steps = 60,
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            # uncomment this to integrate with W&B
            # report_to="none"
        ),
    )
    return (trainer,)


@app.cell
def _(mo):
    mo.md(r"""# Training""")
    return


@app.cell
def _(mo, trainer):
    with mo.persistent_cache(name="trainer_cache"):
        trainer_stats = trainer.train()
    return


@app.cell
def _(FastLanguageModel, model_peft):
    model_for_inference = FastLanguageModel.for_inference(model_peft)
    return (model_for_inference,)


@app.cell
def _():
    input_prompt = """You are an SQL generator that takes the user's query and gives them helpful SQL to use.

    Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. Finally, an explanation justifies the generated.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    SQL:
    """
    prompt = input_prompt.format(
        "Generate an SQL query to obtain all posts with tag 'terminal'. Be minimal.",
        "Company has 3 tables: `posts`, `tags` and `users`."
    )
    return (prompt,)


@app.cell
def _(mo, prompt):
    mo.md(
        rf"""
    #Prompt
    {prompt}
    """
    )
    return


@app.cell
def _(prompt, tokenizer):
    inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")
    return (inputs,)


@app.cell
def _(inputs, mo, model_for_inference, tokenizer):
    with mo.persistent_cache(name="response_cache"):
        outputs = model_for_inference.generate(**inputs, max_new_tokens=100, use_cache=True)
        response = tokenizer.batch_decode(outputs)
    return (response,)


@app.cell
def _(mo, response):
    mo.md(
        rf"""
    #Response from fine tune
    {response[0]}
    """
    )
    return


@app.cell
def _(inputs, mo, model, tokenizer):
    with mo.persistent_cache(name="response_from_original_cache"):
        outputs_from_original = model.generate(**inputs, max_new_tokens=100, use_cache=True)
        response_from_original = tokenizer.batch_decode(outputs_from_original)
    return (response_from_original,)


@app.cell
def _(mo, response_from_original):
    mo.md(
        rf"""
    #Response from oringinal
    {response_from_original[0]}
    """
    )
    return


if __name__ == "__main__":
    app.run()
