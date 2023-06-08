project=byzerllm
version=0.0.5
rm -rf ./dist/*
pip uninstall -y ${project}
python setup.py sdist bdist_wheel
cd ./dist/
pip install ${project}-${version}-py3-none-any.whl && cd -

export MODE=${MODE:-"dev"}

if [[ ${MODE} == "release" ]];then
 twine upload dist/*
fi
