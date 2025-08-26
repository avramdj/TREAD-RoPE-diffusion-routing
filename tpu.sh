#!/bin/bash

# A script to automate the creation of Google Cloud TPU Queued Resources.
# It handles one-time network setup and includes --dry-run, --spot, and custom network options.

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
PROJECT_ID="ml-work-469710"
SUBNET_RANGE="172.16.1.0/24"

# --- Script Functions ---

usage() {
    echo "Usage: $0 <tpu_type> <node_id> <zone> [OPTIONS]"
    echo ""
    echo "Arguments:"
    echo "  <tpu_type>             The type of TPU to create. Supported: 'v4-8', 'v6e-8'."
    echo "  <node_id>              A unique name for the TPU node (e.g., 'my-tpu-v4-node')."
    echo "  <zone>                 The GCP zone for the TPU (e.g., 'us-central2-b')."
    echo ""
    echo "Options:"
    echo "  --dry-run              Optional. Show commands without creating resources."
    echo "  --spot                 Optional. Request a Spot instance instead of on-demand."
    echo "  --network-name <name>  Optional. Specify the network name. Defaults to 'my-network-<zone>'."
    echo "  --subnet-name <name>   Optional. Specify the subnet name. Defaults to 'my-subnet-<zone>'."
    echo ""
    echo "Example:"
    echo "  $0 v4-8 my-awesome-tpu us-central2-b --spot --network-name my-existing-network"
    exit 1
}

# Updated function to handle the dry run logic
setup_network() {
    local network_name=$1
    local subnet_name=$2
    local region=$3
    local is_dry_run=$4 # The fourth argument is our dry run flag

    echo "--- Checking network configuration for region: ${region} ---"

    # Check/Create Network
    if ! gcloud compute networks describe "${network_name}" --project="${PROJECT_ID}" &>/dev/null; then
        local cmd="gcloud compute networks create \"${network_name}\" --project=\"${PROJECT_ID}\" --subnet-mode=custom"
        if [ "$is_dry_run" = true ]; then
            echo "DRY RUN: Network '${network_name}' not found. Would run:"
            echo "  $cmd"
        else
            echo "Network '${network_name}' not found. Creating it..."
            eval "$cmd" # Using eval to correctly execute the command string
        fi
    else
        echo "Network '${network_name}' already exists."
    fi

    # Check/Create Subnet
    if ! gcloud compute networks subnets describe "${subnet_name}" --region="${region}" --project="${PROJECT_ID}" &>/dev/null; then
        local cmd="gcloud compute networks subnets create \"${subnet_name}\" --project=\"${PROJECT_ID}\" --network=\"${network_name}\" --region=\"${region}\" --range=\"${SUBNET_RANGE}\""
        if [ "$is_dry_run" = true ]; then
            echo "DRY RUN: Subnet '${subnet_name}' not found. Would run:"
            echo "  $cmd"
        else
            echo "Subnet '${subnet_name}' in region '${region}' not found. Creating it..."
            eval "$cmd"
        fi
    else
        echo "Subnet '${subnet_name}' already exists in region '${region}'."
    fi

    # Check/Create Firewall Rule
    local firewall_rule_name="allow-ssh-on-${network_name//./-}"
    if ! gcloud compute firewall-rules describe "${firewall_rule_name}" --project="${PROJECT_ID}" &>/dev/null; then
        local cmd="gcloud compute firewall-rules create \"${firewall_rule_name}\" --project=\"${PROJECT_ID}\" --network=\"${network_name}\" --allow=tcp:22"
        if [ "$is_dry_run" = true ]; then
            echo "DRY RUN: Firewall rule '${firewall_rule_name}' not found. Would run:"
            echo "  $cmd"
        else
            echo "Firewall rule '${firewall_rule_name}' not found. Creating it..."
            eval "$cmd"
        fi
    else
        echo "Firewall rule '${firewall_rule_name}' already exists."
    fi
}


# --- Main Script Logic ---

# 1. Argument Parsing for flags
IS_DRY_RUN=false
USE_SPOT=false
CUSTOM_NETWORK_NAME=""
CUSTOM_SUBNET_NAME=""
POS_ARGS=()

while (( "$#" )); do
  case "$1" in
    --dry-run)
      IS_DRY_RUN=true
      shift
      ;;
    --spot)
      USE_SPOT=true
      shift
      ;;
    --network-name)
      if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
        CUSTOM_NETWORK_NAME="$2"
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        usage
      fi
      ;;
    --subnet-name)
      if [ -n "$2" ] && [ "${2:0:1}" != "-" ]; then
        CUSTOM_SUBNET_NAME="$2"
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        usage
      fi
      ;;
    -*) # unsupported flags
      echo "Error: Unsupported flag $1" >&2
      usage
      ;;
    *) # preserve positional arguments
      POS_ARGS+=("$1")
      shift
      ;;
  esac
done
set -- "${POS_ARGS[@]}" # Restore positional arguments

# 2. Validate positional arguments
if [ "$#" -ne 3 ]; then
    usage
fi

TPU_TYPE=$1
NODE_ID=$2
ZONE=$3

# 3. Set up initial variables
if [ -z "$PROJECT_ID" ]; then
    PROJECT_ID=$(gcloud config get-value project)
fi
if [ -z "$PROJECT_ID" ]; then
    echo "Error: Project ID not set. Use 'gcloud config set project <id>' or set it in the script."
    exit 1
fi

REGION="${ZONE%-*}"

# Set network and subnet names: use custom ones if provided, otherwise generate default
if [ -n "$CUSTOM_NETWORK_NAME" ]; then
    NETWORK_NAME="$CUSTOM_NETWORK_NAME"
else
    NETWORK_NAME="my-network"
fi

if [ -n "$CUSTOM_SUBNET_NAME" ]; then
    SUBNET_NAME="$CUSTOM_SUBNET_NAME"
else
    SUBNET_NAME="my-subnet"
fi

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
QUEUED_RESOURCE_ID="${NODE_ID}-qr-${TIMESTAMP}"

# 4. Set TPU-specific configurations
case "$TPU_TYPE" in
    "v4-8") ACCELERATOR_TYPE="v4-8"; RUNTIME_VERSION="tpu-ubuntu2204-base" ;;
    "v6e-8") ACCELERATOR_TYPE="v6e-8"; RUNTIME_VERSION="v2-alpha-tpuv6e" ;;
    *) echo "Error: Unsupported TPU type '${TPU_TYPE}'." >&2; exit 1 ;;
esac

# 5. Print summary and wait for confirmation
echo "--- Configuration Summary ---"
if [ "$IS_DRY_RUN" = true ]; then
    echo "Mode:             DRY RUN (No resources will be created)"
fi
echo "Project:          ${PROJECT_ID}"
echo "TPU Node ID:      ${NODE_ID}"
echo "Zone:             ${ZONE}"
echo "Reservation Type: $(if [ "$USE_SPOT" = true ]; then echo "Spot"; else echo "On-Demand"; fi)"
echo "Accelerator:      ${ACCELERATOR_TYPE}"
echo "Runtime:          ${RUNTIME_VERSION}"
echo "Network Name:     ${NETWORK_NAME}"
echo "Subnet Name:      ${SUBNET_NAME}"
echo "-----------------------------"
read -p "Press Enter to continue or Ctrl+C to cancel..."

# 6. Initial gcloud setup
echo "--- Performing Initial Setup ---"
gcloud config set project "${PROJECT_ID}"
if [ "$IS_DRY_RUN" = false ]; then
    echo "Enabling the TPU API (tpu.googleapis.com)..."
    gcloud services enable tpu.googleapis.com --project="${PROJECT_ID}"
else
    echo "DRY RUN: Would enable the TPU API."
fi

# 7. Set up networking (once per region), passing the dry run flag
setup_network "${NETWORK_NAME}" "${SUBNET_NAME}" "${REGION}" "${IS_DRY_RUN}"

# 8. Build and execute the TPU Queued Resource command
echo "--- Preparing TPU Queued Resource command ---"
GCLOUD_CMD=(
    gcloud compute tpus queued-resources create "${QUEUED_RESOURCE_ID}"
    --project="${PROJECT_ID}"
    --node-id="${NODE_ID}"
    --zone="${ZONE}"
    --accelerator-type="${ACCELERATOR_TYPE}"
    --runtime-version="${RUNTIME_VERSION}"
    --network="${NETWORK_NAME}"
    --subnetwork="${SUBNET_NAME}"
)

# Conditionally add the --spot flag if requested
if [ "$USE_SPOT" = true ]; then
    GCLOUD_CMD+=(--spot)
fi

if [ "$IS_DRY_RUN" = true ]; then
    echo "DRY RUN: Would execute the following command:"
    echo "  ${GCLOUD_CMD[@]}"
else
    echo "--- Creating TPU Queued Resource ---"
    "${GCLOUD_CMD[@]}"
fi

echo "--- Script finished. ---"
if [ "$IS_DRY_RUN" = false ]; then
    echo "TPU Queued Resource '${QUEUED_RESOURCE_ID}' creation command has been submitted."
    echo "Check its status with: gcloud compute tpus queued-resources list --zone=${ZONE}"
    echo ""
    echo "--- Helpful Commands (once the TPU node is ACTIVE) ---"
    echo "# SSH into all TPU workers:"
    echo "gcloud compute tpus tpu-vm ssh ${NODE_ID} --zone=${ZONE} --project=${PROJECT_ID} --worker=all"
    echo ""
    echo "# Run a command on all TPU workers:"
    echo "gcloud compute tpus tpu-vm ssh ${NODE_ID} --zone=${ZONE} --project=${PROJECT_ID} --worker=all --command='sudo apt update'"
    echo ""
    echo "# Copy a file TO all TPU workers:"
    echo "gcloud compute tpus tpu-vm scp /path/to/local/file ${NODE_ID}:/remote/path/ --zone=${ZONE} --project=${PROJECT_ID} --worker=all"
    echo ""
    echo "# Copy a file FROM a TPU worker (e.g., worker 0):"
    echo "gcloud compute tpus tpu-vm scp ${NODE_ID}:/remote/path/file /local/path/ --zone=${ZONE} --project=${PROJECT_ID} --worker=0"
    echo ""
    echo "# Port forward from local machine to worker 0 (e.g., for Jupyter):"
    echo "# Usage: gcloud ... ssh ... -- -L <LOCAL_PORT>:localhost:<REMOTE_PORT>"
    echo "gcloud compute tpus tpu-vm ssh ${NODE_ID} --zone=${ZONE} --project=${PROJECT_ID} --worker=0 -- -L 8888:localhost:8888"
fi
