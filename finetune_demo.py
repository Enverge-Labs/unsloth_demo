import marimo

__generated_with = "0.14.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""# Tudor's first fine tune on Envege Lab""")
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
def _(tokenizer):
    EOS_TOKEN = tokenizer.eos_token

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    Company database: {}

    ### Input:
    SQL Prompt: {}

    ### Response:
    SQL: {}

    Explanation: {}
    """

    def formatting_prompts_func(examples):
        company_databases = examples["sql_context"]
        prompts = examples["sql_prompt"]
        sqls = examples["sql"]
        explanations = examples["sql_explanation"]
        texts = []
        for company_database, prompt, sql, explanation in zip(company_databases, prompts, sqls, explanations):
            text = alpaca_prompt.format(company_database, prompt, sql, explanation) + EOS_TOKEN
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
            # num_train_epochs = 1, # Set this for 1 full training run.
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
        ),
    )
    return (trainer,)


@app.cell
def _(mo):
    mo.md(r"""# Training""")
    return


@app.cell
def _(trainer):
    trainer_stats = trainer.train()
    return


@app.cell
def _(model_peft, tokenizer):
    # skip this cell if the dir already exists
    model_peft.save_pretrained_merged("merged_model", tokenizer, save_method = "merged_16bit",)
    return


@app.cell
def _():
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    return (AutoTokenizer,)


@app.cell
def _():
    from peft import AutoPeftModelForCausalLM
    return (AutoPeftModelForCausalLM,)


@app.cell
def _(AutoPeftModelForCausalLM):
    trained_model = AutoPeftModelForCausalLM.from_pretrained(
        "./merged_model",
    ).to("cuda")
    return (trained_model,)


@app.cell
def _(AutoTokenizer):
    tokenizer_for_inputs = AutoTokenizer.from_pretrained("./merged_model")
    return (tokenizer_for_inputs,)


@app.cell
def _(mo):
    mo.md(
        r"""
    #Prompt
    "Write an SQL statement to query posts with the tag 'terminal'"
    """
    )
    return


@app.cell
def _(tokenizer_for_inputs):
    inputs = tokenizer_for_inputs(
        "Write an SQL statement to query posts with the tag 'terminal'.",
        return_tensors="pt",
    ).to("cuda")
    return (inputs,)


@app.cell
def _(inputs, trained_model):
    # outputs = model_peft.generate(**inputs, max_new_tokens=200)
    outputs = trained_model.generate(**inputs)
    return (outputs,)


@app.cell
def _(outputs, tokenizer_for_inputs):
    response = tokenizer_for_inputs.batch_decode(outputs, skip_special_tokens=True)
    return (response,)


@app.cell
def _(mo, response):
    mo.md(
        rf"""
    # Response
    {response[0]}
    """
    )
    return


if __name__ == "__main__":
    app.run()
