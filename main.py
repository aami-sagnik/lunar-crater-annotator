from src.utils import get_device
from src.dataset import get_datasets, get_titanium_datasets, titanium_collate_fn
from src.utils import get_dataloaders_for, seed_everything
from src.io import load_model_weights, save_model_weights
# from src.models.fpsnet import FPSNet
# from src.models.retinanet import RetinaNet
from src.models.retinanet import RetinaNet
import os

def main():
    DEVICE = get_device()
    N_ATT = 35 # this is an upper bound on the number of craters
    BATCH_SIZE = 12
    EPOCHS = 5
    DIM=1200
    SEED=101
    os.environ["TORCH_HOME"] = os.path.join(os.getcwd(), ".cache", "torch")

    seed_everything(SEED)

    train_dataset, test_dataset = get_datasets("our_dataset", n_att=N_ATT, dim=(DIM, DIM))
    titanium_train_dataset, titanium_test_dataset = get_titanium_datasets("titanium_dataset", dim=(600, 600))
    train_dataloader, test_dataloader = get_dataloaders_for(train_dataset, test_dataset, batch_size=BATCH_SIZE, seed=SEED)
    titanium_train_dataloader, titanium_test_dataloader = get_dataloaders_for(titanium_train_dataset, titanium_test_dataset, batch_size=BATCH_SIZE, collate_fn=titanium_collate_fn, seed=SEED)
    print("Our Train Dataset:", train_dataset)
    print("Our Test Dataset:", test_dataset)
    print("Titanium Train Dataset:", titanium_train_dataset)
    print("Titanium Test Dataset:", titanium_test_dataset)

    # retina_net_state = load_model_weights("saved_weights", "crater_retinanet_weights.pth")

    # model = FPSNet(n_att=N_ATT, n_stuff=0, c_att=50, retina_net_state=retina_net_state, device=DEVICE).to(DEVICE) # instantiating the model
    model = RetinaNet(device=DEVICE)
    # model.train_model(train_dataloader, 0.01, EPOCHS, targets_are_masks=False)
    for i in range(5):
        print(f"Iteration {i+1}")
        print(f"Training on our train dataset... for {EPOCHS} epochs")
        model.train_model(train_dataloader, 0.01, EPOCHS)
        print(f"Training on our test dataset... for {EPOCHS} epochs")
        model.train_model(test_dataloader, 0.01, EPOCHS)
        print(f"Training on titanium train dataset... for {EPOCHS} epochs")
        model.train_model(titanium_train_dataloader, 0.01, EPOCHS, targets_are_masks=False)
        print(f"Training on titanium test dataset... for {EPOCHS} epochs")
        model.train_model(titanium_test_dataloader, 0.01, EPOCHS, targets_are_masks=False)

    test_metrics = model.test_model(test_dataloader)

    print("-- Our Test Metrics --")
    for k, v in test_metrics.items():
        print(f"\"{k}\": {v},")
    
    titanium_test_metrics = model.test_model(titanium_test_dataloader, targets_are_masks=False)

    print("-- Titanium Test Metrics Titanium --")
    for k, v in titanium_test_metrics.items():
        print(f"\"{k}\": {v},")
    
    save_model_weights(model, "saved_weights", "annotator_weights.pth")

if __name__ == "__main__":
    main()
