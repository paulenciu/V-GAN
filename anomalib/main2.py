from anomalib.models import Patchcore
from anomalib.engine import Engine


input_size = (256, 256)
transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = CustomDataset(dataset_type=DatasetType.MVTEC_AD, category="bottle", image_size=(256, 256), transform=transform)
test_dataset = CustomDataset(dataset_type=DatasetType.MVTEC_AD, category="bottle", image_size=(256, 256), train=False, transform=transform)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize components
datamodule = MVTecAD()
model = Patchcore()
engine = Engine()

# Train the model
engine.fit(datamodule=datamodule, model=model)