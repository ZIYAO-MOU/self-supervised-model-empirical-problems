echo "Checking if pip is installed..."
if ! command -v pip &> /dev/null
then
    echo "pip not found. Installing pip..."
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python3 get-pip.py --user
    rm get-pip.py
else
    echo "pip is already installed."
fi

# Step 2: Upgrade pip to the latest version
echo "Upgrading pip..."
python3 -m pip install --upgrade pip

# Step 3: Install packages from requirements.txt
if [ -f "requirements.txt" ]; then
    echo "Installing packages from requirements.txt..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found. Please make sure the file exists in the current directory."
fi

python /home/zmou1/4-RoBERTa/main.py 