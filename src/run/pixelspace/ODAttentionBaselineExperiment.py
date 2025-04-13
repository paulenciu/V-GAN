class ODAttentionBaselineExperiment:

    def __init__(self, dataset_type, category, image_size_od, standardize_data, preprocessing_fn, od_model, root_dir="../experiments/od_baselines"):
        self.dataset_type = dataset_type
        self.category = category
        self.image_size_od = image_size_od
        self.standardize_data = standardize_data
        self.preprocessing_fn = preprocessing_fn
        self.od_model = od_model
        self.root_dir = root_dir

    def fit(self):
        x_train, _ = load_data(dataset_type=self.dataset_type, category=self.category, image_size=self.image_size_od,
                            standardize=self.standardize_data)
        x_train_flattened = extract_and_flatten_images_dataset_3d(x_train).to("cpu").numpy()
        x_train_flattened = self.preprocessing_fn(x_train_flattened)
        self.od_model.fit(x_train_flattened)
        del x_train