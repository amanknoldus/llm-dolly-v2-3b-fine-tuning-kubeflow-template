from datasets import load_dataset
from functools import partial
import numpy as np
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from src.save_model_helper import save_model
from utils.helper_functions import get_log, get_memory_usage, get_model_tokenizer
import pandas as pd
import json
import os

logging = get_log()


def fine_tune_model(dataset_path: str,
                    model_name: str,
                    save_model_bucket_name: str,
                    ):
    try:
        logging.info("Task: Reading training dataset")
        train_df = pd.read_parquet(dataset_path)

        logging.info("Task: Converting csv data to json format")
        training_prompts = train_df.to_dict(orient="records")

        logging.info("Task: Making Directory if not exist for saving training json data")
        save_train_data_path = "./train_data/"
        os.makedirs(save_train_data_path, exist_ok=True)

        logging.info(f"Task: Dumping the training json data to the directory: {save_train_data_path}")
        json_file_path = os.path.join(save_train_data_path, 'train_queries.json')
        with open(json_file_path, 'w') as f:
            json.dump(training_prompts, f)

        logging.info("Task: Reading model and tokenizer")
        """Loading model and tokenizer here"""

        model, tokenizer = get_model_tokenizer(
            pretrained_model_name_or_path=model_name,
            gradient_checkpointing=True
        )

        """ model saving path """
        logging.info("Task: Defining Directory if not exist for saving model")
        local_output_dir = "model_dir"
        os.makedirs(local_output_dir, exist_ok=True)



        """Comment this below code if you do not want to save the default model: line 49 to 54"""

        logging.info(f"Task: Saving the default model & tokenizer to directory: {local_output_dir}")
        model.save_pretrained(local_output_dir)
        logging.info(f"Task: Model saved successfully to the local directory: {local_output_dir}")

        tokenizer.save_pretrained(local_output_dir)
        logging.info(f"Task: Tokenizer saved successfully to the local directory: {local_output_dir}")



        """Uncomment this below code to train model and save it"""

        logging.info("Task: Defining special tokens to be to be used in model training")
        """To be added as special tokens"""
        INSTRUCTION_KEY = "### Instruction:"
        INPUT_KEY = "Input:"
        RESPONSE_KEY = "### Response:"
        END_KEY = "### End"
        RESPONSE_KEY_NL = f"{RESPONSE_KEY}\n"

        logging.info("Task: Getting max length of the model")
        """Find max length in model configuration"""
        max_length = getattr(model.config, "max_position_embeddings", None)

        INTRO_BLURB = (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request."
        )

        logging.info("Task: Defining the parameters if no input format is provided")
        PROMPT_NO_INPUT_FORMAT = """{intro}
        {instruction_key}
        {instruction}
        {response_key}
        {response}
        {end_key}""".format(
            intro=INTRO_BLURB,
            instruction_key=INSTRUCTION_KEY,
            instruction="{instruction}",
            response_key=RESPONSE_KEY,
            response="{response}",
            end_key=END_KEY,
        )

        logging.info("Task: Defining the parameters if input format is provided")
        """training prompt that contains an input string that serves as context"""
        PROMPT_WITH_INPUT_FORMAT = """{intro}
        {instruction_key}
        {instruction}
        {input_key}
        {input}
        {response_key}
        {response}
        {end_key}""".format(
            intro=INTRO_BLURB,
            instruction_key=INSTRUCTION_KEY,
            instruction="{instruction}",
            input_key=INPUT_KEY,
            input="{input}",
            response_key=RESPONSE_KEY,
            response="{response}",
            end_key=END_KEY,
        )

        logging.info("Task: Defining the function to load and process training dataset")
        """Function to load training dataset"""

        def load_training_dataset(path_or_dataset="./train_data/"):
            dataset = load_dataset(path_or_dataset)["train"]

            def _add_text(rec):
                instruction = rec["instruction"]
                response = rec["response"]
                context = rec.get("context")
                if context:
                    rec["text"] = PROMPT_WITH_INPUT_FORMAT.format(
                        instruction=instruction,
                        response=response,
                        input=context
                    )
                else:
                    rec["text"] = PROMPT_NO_INPUT_FORMAT.format(
                        instruction=instruction,
                        response=response
                    )
                return rec

            dataset = dataset.map(_add_text)
            return dataset

        logging.info("Task: Defining the function to process the data into batches")
        """Function to preprocess dataset into batches and tokenize them"""

        def preprocess_batch(batch, get_tokenizer, get_max_length):
            return get_tokenizer(
                batch["text"],
                max_length=get_max_length,
                truncation=True,
            )

        logging.info("Task: Defining the function to map the data")
        """Function to call load dataset and process that."""

        def preprocess_dataset(get_tokenizer, get_max_length):
            dataset = load_training_dataset()
            _preprocessing_function = partial(
                preprocess_batch, max_length=get_max_length, tokenizer=get_tokenizer)
            dataset = dataset.map(
                _preprocessing_function,
                batched=True,
                remove_columns=["instruction", "context", "response", "text"],
            )

            # Make sure we don't have any truncated records, as this would mean the end keyword is missing.
            dataset = dataset.filter(lambda rec: len(rec["input_ids"]) < max_length)
            dataset = dataset.shuffle()
            logging.debug("Memory usage in processing data for training")
            get_memory_usage()
            return dataset

        logging.info("Task: Retrieving the processed data for model training")
        """Retrieving processed dataset"""
        processed_data = preprocess_dataset(tokenizer, max_length)

        logging.info("Task: Defining the function to process data into tokens")
        """Prepare the input data for training a completion-only language model"""

        class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
            def torch_call(self, examples):
                batch = super().torch_call(examples)

                # The prompt ends with the response key plus a newline
                response_token_ids = self.tokenizer.encode(RESPONSE_KEY_NL)
                labels = batch["labels"].clone()

                for i in range(len(examples)):
                    response_token_ids_start_idx = None
                    for idx in np.where(batch["labels"][i] == response_token_ids[0])[0]:
                        response_token_ids_start_idx = idx
                        break

                    if response_token_ids_start_idx is None:
                        raise RuntimeError(
                            f'Could not find response key {response_token_ids} in token IDs {batch["labels"][i]}'
                        )

                    response_token_ids_end_idx = response_token_ids_start_idx + 1

                    # loss function ignore all tokens up through the end of the response key
                    labels[i, :response_token_ids_end_idx] = -100

                batch["labels"] = labels
                get_memory_usage()
                return batch

        logging.info("Task: Getting the tokenized data")
        data_collator = DataCollatorForCompletionOnlyLM(
            tokenizer=tokenizer, mlm=False, return_tensors="pt", pad_to_multiple_of=8
        )

        logging.info("Task: Defining the epoch count for number of iteration of model training")
        """Epoch count to iterate over dataset multiple times"""
        epoch_count = 1

        logging.info("Task: Defining the model training hyperparameters")
        """model training hyperparameter"""
        training_args = TrainingArguments(
            output_dir=local_output_dir,
            learning_rate=5e-05,
            num_train_epochs=epoch_count,
            logging_dir=f"{local_output_dir}/runs",
            logging_strategy="steps",
        )

        logging.info("Task: Defining the parameters for the trainer")
        """setting model training arguments"""
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=processed_data,
            data_collator=data_collator,
        )

        logging.info("Task: Starting the model training")
        """begin model training"""
        trainer.train()

        logging.info("Task: Model training completed successfully")
        logging.debug("Memory usage in model training")
        get_memory_usage()

        logging.info(f"Task: Saving the trained model to directory: {local_output_dir}")
        """save model after training"""
        trainer.save_model(output_dir=local_output_dir)
        logging.info(f"Task: Model saved successfully to the local directory: {local_output_dir}")

        logging.debug("Memory usage in model saving")
        get_memory_usage()

        logging.debug(f"Task: Saving model from local dir to GCS Bucket: {save_model_bucket_name}")
        save_model(save_model_bucket_name, local_output_dir)
        logging.debug(f"Task: Saved model file to GCS Bucket: {save_model_bucket_name} successfully")

    except Exception as e:
        logging.error("Some error occurred in model training component!")
        raise e
