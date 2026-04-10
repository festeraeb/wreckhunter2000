#!/usr/bin/env python3
"""
SonarSniffer License Management System
Handles licensing, trial periods, and license validation
"""

import os
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple


class LicenseManager:
    """Manages SonarSniffer licensing and trial periods"""

    def __init__(self, license_file: str = None):
        self.license_file = license_file or self._get_default_license_file()
        self.license_data = self._load_license()

    def _get_default_license_file(self) -> str:
        """Get the default license file path"""
        # Use user's home directory for license storage
        home_dir = Path.home()
        license_dir = home_dir / ".sonarsniffer"
        license_dir.mkdir(exist_ok=True)
        return str(license_dir / "license.json")

    def _load_license(self) -> Dict:
        """Load license data from file"""
        if os.path.exists(self.license_file):
            try:
                with open(self.license_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        # Return default trial license
        return self._create_trial_license()

    def _create_trial_license(self) -> Dict:
        """Create a new trial license"""
        install_date = datetime.now()
        expiry_date = install_date + timedelta(days=30)

        license_data = {
            "license_type": "trial",
            "install_date": install_date.isoformat(),
            "expiry_date": expiry_date.isoformat(),
            "features_enabled": True,
            "commercial_use": False,
            "search_rescue": False,
            "license_key": None,
            "contact_email": "festeraeb@yahoo.com",
        }

        # Save the trial license
        self._save_license(license_data)
        return license_data

    def _save_license(self, license_data: Dict):
        """Save license data to file"""
        try:
            with open(self.license_file, "w") as f:
                json.dump(license_data, f, indent=2)
        except IOError:
            pass  # Silently fail if we can't save

    def is_license_valid(self) -> Tuple[bool, str]:
        """
        Check if the current license is valid
        Returns: (is_valid, message)
        """
        try:
            now = datetime.now(datetime.now().astimezone().tzinfo)
        except:
            now = datetime.now()

        # Check expiry date
        if "expiry_date" in self.license_data:
            try:
                expiry = datetime.fromisoformat(self.license_data["expiry_date"])
                # Make both naive for comparison
                if expiry.tzinfo is not None:
                    expiry = expiry.replace(tzinfo=None)
                if now.tzinfo is not None:
                    now = now.replace(tzinfo=None)
                if now > expiry:
                    return (
                        False,
                        "Trial license expired. Contact festeraeb@yahoo.com for a commercial license.",
                    )
            except Exception as e:
                # If date parsing fails, assume valid
                pass

        # Check license type
        license_type = self.license_data.get("license_type", "trial")

        if license_type == "trial":
            days_remaining = self.get_days_remaining()
            if days_remaining <= 0:
                return (
                    False,
                    "Trial license expired. Contact festeraeb@yahoo.com for a commercial license.",
                )
            elif days_remaining <= 7:
                return (
                    True,
                    f"Trial license expires in {days_remaining} days. Contact festeraeb@yahoo.com for a commercial license.",
                )

        elif license_type == "commercial":
            return True, "Commercial license active."

        elif license_type == "search_rescue":
            return True, "Search & Rescue license active."

        return True, "License valid."

    def get_days_remaining(self) -> int:
        """Get days remaining in trial period"""
        if "expiry_date" not in self.license_data:
            return 30

        expiry = datetime.fromisoformat(self.license_data["expiry_date"])
        now = datetime.now(expiry.tzinfo) if expiry.tzinfo else datetime.now()
        remaining = expiry - now
        return max(0, remaining.days)

    def activate_commercial_license(self, license_key: str) -> bool:
        """Activate a commercial license"""
        # In a real implementation, this would validate the license key
        # For now, we'll accept any non-empty key
        if not license_key or len(license_key.strip()) < 10:
            return False

        self.license_data.update(
            {
                "license_type": "commercial",
                "license_key": license_key,
                "activated_date": datetime.now().isoformat(),
                "commercial_use": True,
                "features_enabled": True,
            }
        )

        self._save_license(self.license_data)
        return True

    def activate_sar_license(self, organization: str) -> bool:
        """Activate a Search & Rescue license"""
        if not organization or len(organization.strip()) < 3:
            return False

        self.license_data.update(
            {
                "license_type": "search_rescue",
                "organization": organization,
                "activated_date": datetime.now().isoformat(),
                "search_rescue": True,
                "commercial_use": False,
                "features_enabled": True,
            }
        )

        self._save_license(self.license_data)
        return True

    def get_license_info(self) -> Dict:
        """Get current license information"""
        info = self.license_data.copy()
        info["days_remaining"] = self.get_days_remaining()
        valid, message = self.is_license_valid()
        info["is_valid"] = valid
        info["status_message"] = message
        return info

    def show_license_dialog(self):
        """Display license information dialog"""
        info = self.get_license_info()

        print("\n" + "=" * 60)
        print("🧭 SONARSNIFFER LICENSE INFORMATION")
        print("=" * 60)

        license_type = info.get("license_type", "trial").title()
        print(f"License Type: {license_type}")

        if info.get("license_type") == "trial":
            days = info.get("days_remaining", 0)
            print(f"Days Remaining: {days}")
            if days <= 7:
                print("⚠️  TRIAL EXPIRING SOON!")

        elif info.get("license_type") == "commercial":
            print("Status: Active (Commercial)")

        elif info.get("license_type") == "search_rescue":
            org = info.get("organization", "Unknown")
            print(f"Organization: {org}")
            print("Status: Active (Search & Rescue - FREE)")

        print(f"Status: {'Valid' if info.get('is_valid', False) else 'Invalid'}")
        print(f"Message: {info.get('status_message', '')}")

        print("\n📧 CONTACT INFORMATION:")
        print("Email: festeraeb@yahoo.com")
        print("Subject: 'SonarSniffer License Inquiry'")
        print("SAR organizations: 'SAR Free License Request'")

        print("=" * 60 + "\n")

        return info


# Global license manager instance
_license_manager = None


def get_license_manager() -> LicenseManager:
    """Get the global license manager instance"""
    global _license_manager
    if _license_manager is None:
        _license_manager = LicenseManager()
    return _license_manager


def check_license() -> bool:
    """Check if license is valid, show dialog if needed"""
    manager = get_license_manager()
    valid, message = manager.is_license_valid()

    if not valid or "expires in" in message.lower():
        manager.show_license_dialog()
        return valid

    return True


def require_license():
    """Require valid license, exit if invalid"""
    if not check_license():
        print("❌ License validation failed. Exiting.")
        print("📧 Contact festeraeb@yahoo.com for licensing information.")
        exit(1)


if __name__ == "__main__":
    # Test the license system
    manager = LicenseManager()
    manager.show_license_dialog()
