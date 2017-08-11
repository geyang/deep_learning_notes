from .waterbear import Bear


def test():
    test_dict = {
        'a': 0,
        'b': 1
    }

    test_args = Bear(**test_dict)
    assert test_args.a == 0
    assert test_args.b == 1
    test_args.haha = 0
    assert test_args.haha == 0
    test_args.haha = {'a': 1}
    assert test_args.haha != {'a': 1}
    assert vars(test_args.haha) == {'a': 1}
    assert test_args.haha.a == 1
    assert test_args.__dict__['haha']['a'] == 1
    assert vars(test_args)['haha']['a'] == 1
    assert str(test_args) == "{'a': 0, 'b': 1, 'haha': {'a': 1}}", \
        'test_args should be this value "{\'a\': 0, \'b\': 1, \'haha\': {\'a\': 1}}"'

    test_args = Bear(__recursive=False, **test_dict)
    assert test_args.__is_recursive == False
    assert test_args.a == 0
    assert test_args.b == 1
    test_args.haha = {'a': 1}
    assert test_args.haha['a'] == 1
    assert test_args.haha == {'a': 1}

    # Some other usage patterns
    test_args = Bear(**test_dict, **{'ha': 'ha', 'no': 'no'})
    assert test_args.ha == 'ha', 'key ha should be ha'
