from datasets import load_dataset

from im2latex.evaluators.ocrevaluator import OCREvaluator


def evaluate_im2latex():
    models = {
        "Im2LaTeX": "Matthijs0/im2latex",
        # "Im2LaTeX_ref": "DGurgurov/im2latex"
    }

    dataset = load_dataset("OleehyO/latex-formulas", "cleaned_formulas")
    # dataset = load_dataset("linxy/LaTeX_OCR", "synthetic_handwrite")
    train_val_split = dataset["train"].train_test_split(test_size=0.2,
                                                        seed=42)
    train_ds = train_val_split["train"]
    val_test_split = train_val_split["test"].train_test_split(test_size=0.5, seed=42)
    val_ds = val_test_split["train"]
    test_ds = val_test_split["test"]

    evaluator = OCREvaluator(models, raw_dataset=test_ds, batch_size=12)
    evaluator.evaluate(use_grad_cam=False, grad_cam_batches=2)


# For simplicity, I am making this a separate script, as I want
# to not have this be dependent on a config file.
if __name__ == "__main__":
    evaluate_im2latex()
