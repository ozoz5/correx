import shutil
import subprocess  # nosec B404
import sys

SERVICE_NAME = "correx"


def supports_secure_storage():
    return sys.platform == "darwin" and shutil.which("security") is not None


def get_secure_secret(account_name):
    if not supports_secure_storage():
        return None

    try:
        result = subprocess.run(  # nosec B607
            [
                "security",
                "find-generic-password",
                "-s",
                SERVICE_NAME,
                "-a",
                account_name,
                "-w",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return None

    return result.stdout.strip() or None


def set_secure_secret(account_name, secret_value):
    if not supports_secure_storage():
        return False

    try:
        subprocess.run(  # nosec B607
            [
                "security",
                "add-generic-password",
                "-U",
                "-s",
                SERVICE_NAME,
                "-a",
                account_name,
                "-w",
                secret_value,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def delete_secure_secret(account_name):
    if not supports_secure_storage():
        return False

    try:
        subprocess.run(  # nosec B607
            [
                "security",
                "delete-generic-password",
                "-s",
                SERVICE_NAME,
                "-a",
                account_name,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except subprocess.CalledProcessError as error:
        stderr = (error.stderr or "").lower()
        if "could not be found" in stderr or error.returncode == 44:
            return True
        return False
