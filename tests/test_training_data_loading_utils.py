from src.utils.data_loading import dataloader_runtime_kwargs


def test_dataloader_runtime_kwargs_omits_worker_only_options_without_workers():
    kwargs = dataloader_runtime_kwargs(num_workers=0, pin_memory=True, prefetch_factor=4)

    assert kwargs == {"num_workers": 0, "pin_memory": True}


def test_dataloader_runtime_kwargs_enables_persistent_workers_when_workers_exist():
    kwargs = dataloader_runtime_kwargs(num_workers=2, pin_memory=False, prefetch_factor=4)

    assert kwargs["num_workers"] == 2
    assert kwargs["pin_memory"] is False
    assert kwargs["persistent_workers"] is True
    assert kwargs["prefetch_factor"] == 4
