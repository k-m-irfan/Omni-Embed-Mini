"""Monkey-patches for mteb 2.12.13 bugs encountered on MAEB.

Imported at the top of run_mteb.py; the patches are applied on import.

Bug fixed: `dataset_transform() got unexpected keyword argument 'num_proc'`

Root cause: in mteb 2.12, `AbsTask.load_data()` calls
`self.dataset_transform(num_proc=num_proc)` but several audio-task subclasses
override `dataset_transform(self)` without `**kwargs`. Upstream main has the
same signature, so this is an upstream regression. Patched here.
"""
from __future__ import annotations


def _wrap_no_kwargs(cls):
    """Rewrap cls.dataset_transform so it ignores extra kwargs."""
    orig = cls.dataset_transform

    def shim(self, *args, **kwargs):  # noqa: ARG001  (swallow kwargs)
        return orig(self)

    cls.dataset_transform = shim


_patched = False


def apply():
    global _patched
    if _patched:
        return
    import mteb.tasks.classification.eng.common_language_age_detection as _m1
    import mteb.tasks.classification.eng.iemocap_gender as _m2
    import mteb.tasks.classification.eng.vox_celeb_sa as _m3
    import mteb.tasks.clustering.eng.crema_d_clustering as _m4

    for mod, name in (
        (_m1, "CommonLanguageAgeDetection"),
        (_m2, "IEMOCAPGenderClassification"),
        (_m3, "VoxCelebSA"),
        (_m4, "CREMADClustering"),
    ):
        cls = getattr(mod, name, None)
        if cls is not None and "num_proc" not in cls.dataset_transform.__code__.co_varnames:
            _wrap_no_kwargs(cls)
    _patched = True
    print("[patch_mteb] patched 4 task classes: CommonLanguageAgeDetection, "
          "IEMOCAPGenderClassification, VoxCelebSA, CREMADClustering")


def _patch_datasets_numpy_key():
    """Cast numpy integer indices to Python int inside datasets' key-type check.

    FSD2019Kaggle (AudioMultilabelClassification) and
    VoxPopuliAccentPairClassification iterate with numpy.int64 and pass it to
    `Dataset.__getitem__`, which raises `TypeError: Wrong key type … numpy.int64`
    on the pinned datasets version.
    """
    import numpy as np
    from datasets.formatting import formatting as _f

    orig_ft = _f.format_table

    def ft_shim(table, key, formatter, format_columns=None, output_all_columns=False):
        if isinstance(key, np.integer):
            key = int(key)
        return orig_ft(table, key, formatter, format_columns, output_all_columns)

    _f.format_table = ft_shim
    # arrow_dataset imports format_table by name — rebind there too.
    from datasets import arrow_dataset as _ad
    _ad.format_table = ft_shim
    print("[patch_mteb] datasets.format_table numpy-int key-cast shim installed")


apply()
_patch_datasets_numpy_key()
