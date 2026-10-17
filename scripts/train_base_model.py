"""Generate the base world-model checkpoint if it is missing."""

from run_pipeline import DATASETS, collect_experience, train_world_model

MODEL_PATH = DATASETS / "world_model.pkl"


def main():
    if MODEL_PATH.is_file():
        print(f"[base-model] Existing checkpoint: {MODEL_PATH}")
        return

    print("[base-model] Checkpoint missing; collecting experience...")
    S, A, SN, tracks = collect_experience()

    print("[base-model] Training robot and obstacle models...")
    train_world_model(S, A, SN, tracks)

    if not MODEL_PATH.is_file():
        raise RuntimeError("Training did not create world_model.pkl")

    print("[base-model] Checkpoint ready.")


if __name__ == "__main__":
    main()
