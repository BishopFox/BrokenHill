import os

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            return line.split("'")[1]

    raise RuntimeError('Unable to find version string.')


with open('requirements.txt', 'r') as requirements:
    base_package_list = find_packages()
    packages_that_setuptools_cannot_find_for_some_reason = [ "llm_attacks_bishopfox.jailbreak_detection" ]
    for p in packages_that_setuptools_cannot_find_for_some_reason:
       if p not in base_package_list:
           base_package_list.append(p)
    setup(name='llm_attacks_bishopfox',
          version=get_version('llm_attacks_bishopfox/__init__.py'),
          install_requires=list(requirements.read().splitlines()),
          packages=base_package_list,
          package_data={'': ['*.json']},
          include_package_data=True,
          description='library and tooling for creating adversarial prompts for language models',
          python_requires='>=3.6',
          author='Andy Zou, Zifan Wang, Matt Fredrikson, J. Zico Kolter, Ben Lincoln',
          #author_email='jzou4@andrew.cmu.edu',
          classifiers=[
              'Programming Language :: Python :: 3',
              'License :: OSI Approved :: MIT License',
              'Operating System :: OS Independent'
          ],
          long_description=long_description,
          long_description_content_type='text/markdown')