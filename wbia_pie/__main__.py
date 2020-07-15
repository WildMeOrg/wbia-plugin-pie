# -*- coding: utf-8 -*-
def main():  # nocover
    import wbia_pie

    print('Looks like the imports worked')
    print('wbia_pie = {!r}'.format(wbia_pie))
    print('wbia_pie.__file__ = {!r}'.format(wbia_pie.__file__))
    print('wbia_pie.__version__ = {!r}'.format(wbia_pie.__version__))


if __name__ == '__main__':
    """
    CommandLine:
       python -m wbia_pie
    """
    main()
