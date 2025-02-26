class OutlierDetectionExperiment:

    def __init__(self, vmmd, od_model, dataset_type, category, image_size_generator, standardize_data=False, image_size_od=image_size_generator):
        self.vmmd = vmmd
        self.vmmd_od = VMMDOD(vmmd)
        self.vmmd_wrapper = VMMDWrapper(vmmd)
        self.dataset_type = dataset_type
        self.category = category
        self.image_size_generator = image_size_generator
        self.image_size_od = image_size_od
        self.standardize_data = standardize_data
        self.od_model = od_model

    def run(self, store_stats=True):
        x_train = load_data(dataset_type=self.dataset_type, category=self.category, image_size=self.image_size_train, standardize=self.standardize_data)
        self.vmmd.fit(dataset=x_train, encoder=self.encoder, generator=self.generator, preprocess_fn=self.preprocessing_fn)
        del x_train

        self.vmmd.seed = self.seed
        self.vmmd.approx_subspace_dist(subspace_count=self.subspace_count)
        subspaces = self.vmmd.subspaces

        #Preparing subspaces for OD
        subspaces = np.array(
            interpolate(subspaces, (subspaces.shape[0], subspaces.shape[1], self.image_size_od, self.image_size_od)),
            dtype=int)
        unique_subspace_count = len(subspaces)
        print("Number of unique subspaces:", unique_subspace_count, "/", sample_subspace_count)

        x_train = load_data(dataset_type=self.dataset_type, category=self.category, image_size=self.image_size_od, standardize=self.standardize_data)
        self.od_model.fit(subspaces, x_train)
        del x_train

        x_test, y_test = load_data(dataset_type=self.dataset_type, category=self.category, image_size=self.image_size_od, standardize=self.standardize_data, train=False)
        decision_scores = self.od_model.decistion_function(x_test)

        od_stats = self.calculate_od_stats(y_test, decision_scores)
        return od_stats if not store_stats else self.vmmd_od.store_od_stats(stats, run_number=-1)


    def calculate_od_stats(self, y_test, decision_scores):
        return {"Dataset": self.dataset_type,
                "AUC": auc(y_test, decision_scores),
                "PRAUC": average_precision_score(y_test, decision_scores),
                "F1": f1_score(y_test, (decision_scores > np.quantile(decision_scores, .80)) * 1),
                "Training Time": str(datetime.timedelta(seconds=self.od_model.fit_time)),
                "Decision Time": str(datetime.timedelta(seconds=self.od_model.decision_time))}



