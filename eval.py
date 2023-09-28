from scidocs import get_scidocs_metrics
from scidocs.paths import DataPaths

# point to the data, which should be in scidocs/data by default
data_paths = DataPaths('../data/scidocs/')

# point to the included embeddings jsonl
classification_embeddings_path = '../output/cls.jsonl'
user_activity_and_citations_embeddings_path = '../output/user-citation.jsonl'

# now run the evaluation
scidocs_metrics = get_scidocs_metrics(
    data_paths,
    classification_embeddings_path,
    user_activity_and_citations_embeddings_path,
    val_or_test='test',  # set to 'val' if tuning hyperparams
    n_jobs=12  # the classification tasks can be parallelized
)

print(scidocs_metrics)
