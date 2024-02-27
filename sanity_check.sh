#!/bin/bash

# * all inputs *

# all color codes
GREEN='\033[0;32m'
RED='\033[0;31m' 
NC='\033[0m' # No Color

# Input definitions to ensure we can change in the future if needed 

# 1. HugePages 
HUGEPAGES_CHECK_GREP='hugepagesz=1G'
HUGEPAGES_CHECK_FSTAB='hugetlbfs /dev/hugepages-1G'

# 2. pci driver
LSMOD_GREP='tenstorrent'

# 3. tt-flash
RUSTUP_COMMAND='rustup'
TT_FLASH_COMMAND='tt-flash'
TT_FLASH_HELP_ARG='--help'

# 4. pcks dep
REQUIRED_PACKAGES=("software-properties-common" "python3.8-venv" "libboost-all-dev" "libgoogle-glog-dev" "libgl1-mesa-glx" "libyaml-cpp-dev" "ruby" "build-essential" "clang-6.0" "libhdf5-dev" "libzmq3-dev")

# 5. tt-smi
TT_SMI_COMMAND='tt-smi'
TT_SMI_HELP_ARG='--help'

# 6. tt-flash
TT_FLASH_COMMAND='tt-flash'
TT_FLASH_HELP_ARG='--help'

# 7. pybuda
PYTHON_COMMAND='python3'
PYBUDA_IMPORT_CHECK='import pybuda'

# Progress bar
TOTAL_STEPS=7
CURRENT_STEP=0

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Function to check if a command is available
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

print_progress() {
    PERCENT=$((100 * CURRENT_STEP / TOTAL_STEPS))
    printf "Progress: %s%% [%s%s]\n" "$PERCENT" "$(seq -s "#" $((PERCENT / 2)) | tr -d '[:digit:]')" "$(seq -s " " $(((100 - PERCENT) / 2)) | tr -d '[:digit:]')"
}

# 1. hugepages
grep -q "$HUGEPAGES_CHECK_GREP" /etc/default/grub && \
grep -q "$HUGEPAGES_CHECK_FSTAB" /etc/fstab

if [ $? -eq 0 ]; then
    echo -e "${GREEN}1. Setup HugePages: OK${NC}"
else
    echo -e "${RED}1. Setup HugePages: Not properly configured${NC}"
fi
CURRENT_STEP=$((CURRENT_STEP + 1))
print_progress

# 2. pci driver
if command_exists lsmod && lsmod | grep "$LSMOD_GREP" >/dev/null 2>&1; then
    echo -e "${GREEN}2. PCI Driver Installation: OK${NC}"
else
    echo -e "${RED}2. PCI Driver Installation: Not installed${NC}"
fi
CURRENT_STEP=$((CURRENT_STEP + 1))
print_progress

# 3. tt-flash check
if command_exists "$RUSTUP_COMMAND" && \
   command_exists "$TT_FLASH_COMMAND" && "$TT_FLASH_COMMAND" "$TT_FLASH_HELP_ARG" >/dev/null 2>&1; then
    echo -e "${GREEN}3. Device Firmware Update: OK${NC}"
else
    echo -e "${RED}3. Device Firmware Update: Not installed${NC}"
fi
CURRENT_STEP=$((CURRENT_STEP + 1))
print_progress

# 4. pcks dep
missing_packages=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! apt list --installed 2>/dev/null | grep -q "^$package/"; then
        missing_packages+=("$package")
    fi
done

if [ ${#missing_packages[@]} -eq 0 ]; then
    echo -e "${GREEN}4. Packages are installed correctly${NC}"
else
    echo -e "${RED}4. Packages are not installed correctly${NC}"
    echo -e "${RED}Missing packages: ${missing_packages[@]}${NC}"
fi
CURRENT_STEP=$((CURRENT_STEP + 1))
print_progress

# 5. TT-SMI check
if command_exists "$TT_SMI_COMMAND" && "$TT_SMI_COMMAND" "$TT_SMI_HELP_ARG" >/dev/null 2>&1; then
    echo -e "${GREEN}5. TT-SMI Installation: OK${NC}"
else
    echo -e "${RED}5. TT-SMI Installation: Not installed${NC}"
fi
CURRENT_STEP=$((CURRENT_STEP + 1))
print_progress

# 6. TT-Flash check
if command_exists "$TT_FLASH_COMMAND" && "$TT_FLASH_COMMAND" "$TT_FLASH_HELP_ARG" >/dev/null 2>&1; then
    echo -e "${GREEN}6. TT-Flash Installation: OK${NC}"
else
    echo -e "${RED}6. TT-Flash Installation: Not installed${NC}"
fi
CURRENT_STEP=$((CURRENT_STEP + 1))
print_progress

# 7. pybuda check 
if command_exists "$PYTHON_COMMAND" && "$PYTHON_COMMAND" -c "$PYBUDA_IMPORT_CHECK" >/dev/null 2>&1; then
    echo -e "${GREEN}7. PyBuda installation : OK${NC}"
else
    echo -e "${RED}7. PyBuda Installation: Not installed or import failed${NC}"
fi
CURRENT_STEP=$((CURRENT_STEP + 1))
print_progress

# print out results 
if [ $CURRENT_STEP -eq $TOTAL_STEPS ]; then
    echo -e "${GREEN}All Installs Good :)${NC}"
else
    echo -e "${RED}2 Check Installs Step(s) in Red :(${NC}"
fi