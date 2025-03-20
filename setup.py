import setuptools

setuptools.setup(
    name = "FateExplorer",
    version = "v0.0.1",
    packages = setuptools.find_packages(),
    # Metadata
    url = "https://github.com/jiang-junyao/FateExplorer",
    author = "Junyao Jiang",
    author_email = "jyjiang@link.cuhk.edu.hk",
    description = "Word2Vec based clone representation \
            and fate analysis",
    python_requires = ">=3.9",
    include_package_data = True,
    package_data = {'': [
        'data/*'
    ]}
    #...
)