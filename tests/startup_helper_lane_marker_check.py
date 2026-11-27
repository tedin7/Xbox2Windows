from startup_gfx2hlsl_dpp_whole_quad_window_check import READ_V3, base, pipeline


def main():
    helper, reason = pipeline(base(read=READ_V3))
    assert helper is not None, reason
    assert helper.ps_profile() == 'ps_6_6', helper.ps_profile()
    assert helper.manifest().splitlines().count('helper_lane') == 1, helper.manifest()

    plain, reason = pipeline(base())
    assert plain is not None, reason
    assert 'helper_lane' not in plain.manifest().splitlines(), plain.manifest()
    print('helper_lane manifest marker and ps_6_6 profile: ok')


if __name__ == '__main__':
    main()
