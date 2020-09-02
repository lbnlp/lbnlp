import os
from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()


module_dir = os.path.dirname(os.path.abspath(__file__))

# Requirements
reqs_file = os.path.join(module_dir, "requirements.txt")
with open(reqs_file, "r") as f:
    reqs_raw = f.read()
reqs_list = [r.replace("==", ">=") for r in reqs_raw.split("\n")]

# Optional requirements
extras_file = os.path.join(module_dir, "requirements-optional.txt")
with open(extras_file, "r") as f:
    extras_raw = f.read()
extras_raw = [r for r in extras_raw.split("##") if r.strip() and "#" not in r]
extras_dict = {}
for req in extras_raw:
    items = [i.replace("==", ">=") for i in req.split("\n") if i.strip()]
    dependency_name = items[0].strip()
    dependency_reqs = [i.strip() for i in items[1:] if i.strip()]
    extras_dict[dependency_name] = dependency_reqs
extras_list = [r for d in extras_dict.values() for r in d]

setup(
    name='lbnlp',
    version='0.1',
    description='Common text mining tools for materials data',
    long_description=open(os.path.join(module_dir, 'README.md')).read(),
    url='https://github.com/materialsintelligence/lbnlp',
    author='Alex Dunn',
    author_email='ardunn@lbl.gov',
    license='modified BSD',
    packages=find_packages(),
    package_data={},
    zip_safe=False,
    install_requires=reqs_list,
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
    tests_require=extras_list,
    scripts=[]
)
