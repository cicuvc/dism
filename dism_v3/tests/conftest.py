import pytest
import cu_flash_dism as cu


def pytest_collection_modifyitems(items):
    disabled = pytest.mark.skip(reason='optional FP32 validation instances are not compiled')
    probes_disabled = pytest.mark.skip(reason='requires DISM_BUILD_PROBES=1')
    probe_files = {
        'test_allocator.py','test_key_metadata.py','test_summary_sync.py',
        'test_forward_scan.py','test_backward_scan.py','test_output_store.py',
        'test_forward_boundaries.py','test_backward_summary.py','test_backward_qk.py',
        'test_backward_components.py','test_backward_acceptance.py','test_backward_key_store.py',
        'test_backward_chunk.py',
    }
    probe_tests = {
        'test_varlen_layout.py': {'test_packing','test_lowlevel_checks',
                                 'test_device_descriptor_array','test_scalar_unpack'},
        'test_summary.py': {'test_causal_mask_does_not_affect_valid_diagonals',
                            'test_finite_masked_scan'},
    }
    for item in items:
        callspec = getattr(item, 'callspec', None)
        if not cu.fp32_enabled() and callspec and callspec.params.get('fp32_output') is True:
            item.add_marker(disabled)
        name=item.originalname or item.name
        if not hasattr(cu,'scan_probe') and (item.path.name in probe_files
                or name in probe_tests.get(item.path.name,set())
                or (item.path.name=='test_summary.py' and 'probe' in name)):
            item.add_marker(probes_disabled)
