echo "Y" | pip uninstall gym_pybullet_adrp
rm -rf dist/
poetry build 
pip install dist/gym_pybullet_adrp-1.0.0-py3-none-any.whl
cd tests
python test_build.py
rm -rf results
cd ..