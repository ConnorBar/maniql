
echo """
Setting up the environment!

"""

conda create --name tacsl python==3.8 -y
conda activate tacsl


echo """
Download TacSL-specific Isaac Gym binary from the shared drive:

https://purdue0-my.sharepoint.com/:f:/r/personal/tamosa_purdue_edu/Documents/Marslab%20Capstone%20Project/Codebase/IsaacGym_Preview_TacSL_Package?csf=1&web=1&e=OasHLX
"""

while true; do
  read -p "downloaded? (y/n): " yn
  case $yn in
    [Yy]* )
      if [ -d "$DATA_DIR" ]; then
        echo "Awesome. Continuing setup..."
        break
      else
        echo "bruh go download it u lying"
      fi
      ;;
    [Nn]* )
      echo "go download it"
      ;;
    *) echo "Please y or n";;
  esac
done

pip install -e IsaacGym_Preview_TacSL_Package/isaacgym/python/


echo """
Downloading manifeel github and installing in env
"""
git clone git@github.com:quan-luu/manifeel-isaacgymenvs.git
cd manifeel-isaacgymenvs

git checkout -b tacsl-manifeel-rl

pip install -e .
pip install -r blob/manifeel-tacff/isaacgymenvs/tacsl_sensors/install/requirements.txt





