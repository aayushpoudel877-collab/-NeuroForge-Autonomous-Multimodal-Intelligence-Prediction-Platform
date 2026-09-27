from neuroforge.deep_learning.runs import RunConfig,run_id,TrainingRun,RunStore
from neuroforge.deep_learning.search import SearchSpace,grid_configs
from neuroforge.deep_learning.selection import select_best

def test_run_id_is_deterministic():
    config=RunConfig()
    assert run_id(config)==run_id(config)

def test_grid_search_count():
    configs=list(grid_configs(SearchSpace(learning_rates=(.001,.0003),batch_sizes=(16,32),epochs=(5,10))))
    assert len(configs)==8

def test_best_run_selection():
    runs=[
        {"status":"completed","metrics":{"f1":.61}},
        {"status":"completed","metrics":{"f1":.84}},
        {"status":"running","metrics":{"f1":.99}},
    ]
    assert select_best(runs)["metrics"]["f1"]==.84
