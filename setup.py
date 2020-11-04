import os
from setuptools import setup, find_packages
from pip._internal.req import parse_requirements
from pip._internal.network.session import PipSession

this_dir = os.path.dirname(os.path.abspath(__file__))
pip_requirements = parse_requirements(
    os.path.join(this_dir, "requirements.txt"), PipSession())

pip_requirements_ner = parse_requirements(
    os.path.join(this_dir, "requirements-ner.txt"), PipSession())
reqs = [pii.requirement for pii in pip_requirements]
reqs_ner = [pii.requirement for pii in pip_requirements_ner]


extras_dict = {
    "ner": reqs_ner
}

readme_path = os.path.join(this_dir, "README.md")

with open(readme_path, "r") as f:
    long_description = f.read()

setup(
    name='lbnlp',
    version='0.1',
    description='Common text mining tools for materials data',
    long_description=long_description,
    url='https://github.com/materialsintelligence/lbnlp',
    author='Alex Dunn',
    author_email='ardunn@lbl.gov',
    license='modified BSD',
    packages=find_packages(),
    package_data={
        "lbnlp.models": ["*.json"]
    },
    zip_safe=False,
    install_requires=reqs,
    extras_require=extras_dict,
    classifiers=['Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3.6',
                 'Development Status :: 4 - Beta',
                 'Intended Audience :: Science/Research',
                 'Intended Audience :: System Administrators',
                 'Intended Audience :: Information Technology',
                 'Operating System :: OS Independent',
                 'Topic :: Other/Nonlisted Topic',
                 'Topic :: Scientific/Engineering'],
    test_suite='lbnlp',
    # tests_require=extras_list,
    scripts=[]
)
