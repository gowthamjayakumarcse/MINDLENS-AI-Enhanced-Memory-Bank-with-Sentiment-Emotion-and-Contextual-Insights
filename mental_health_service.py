#!/usr/bin/env python3
"""
Mental Health Support Service using OpenStreetMap (No API Key Required)
"""

import requests
from typing import List, Dict

class MentalHealthService:
    """Service to find nearby mental health hospitals using OpenStreetMap."""
    
    def __init__(self):
        """Initialize the service."""
        self.user_agent = "MindLens-mental-health-finder/1.0"
    
    def clean_address(self, address: str) -> str:
        """Auto-correct address formatting for Nominatim."""
        addr = address.replace(",", ", ").replace("  ", " ").strip()
        addr = addr.title()
        if "India" not in addr:
            addr += ", India"
        return addr

    def geocode_location(self, place_name: str) -> tuple:
        """Convert place name to latitude, longitude using OSM Nominatim with fallback."""
        print(f"📍 Getting coordinates for '{place_name}' ...")
        url = "https://nominatim.openstreetmap.org/search"
        # Normalize/clean address
        cleaned = self.clean_address(place_name)
        params = {"q": cleaned, "format": "json", "limit": 1}
        headers = {"User-Agent": self.user_agent}
        
        try:
            res = requests.get(url, params=params, headers=headers, timeout=30)
            res.raise_for_status()
            data = res.json()
            
            # Fallback: broaden search (e.g., district,state,country)
            if not data:
                parts = cleaned.split(",")
                if len(parts) > 2:
                    retry_query = ", ".join([p.strip() for p in parts[-3:]])
                    print(f"⚠️ Retrying with broader location: {retry_query}")
                    params["q"] = retry_query
                    res = requests.get(url, params=params, headers=headers, timeout=30)
                    res.raise_for_status()
                    data = res.json()
            
            if not data:
                raise ValueError(f"❌ Could not find location for '{place_name}'")
            
            lat = float(data[0]["lat"])
            lon = float(data[0]["lon"])
            print(f"✅ Location found: {lat}, {lon}")
            return lat, lon
        except Exception as e:
            print(f"❌ Geocoding error: {e}")
            raise
    
    def find_nearby_hospitals(self, place_name: str, radius_m: int = 10000) -> List[Dict[str, str]]:
        """Find general hospitals within the given radius using Overpass API."""
        try:
            lat, lon = self.geocode_location(place_name)
            print(f"🔍 Searching for hospitals within {radius_m/1000:.1f} km...\n")
            
            overpass_url = "https://overpass.kumi.systems/api/interpreter"
            query = f"""
            [out:json];
            node(around:{radius_m},{lat},{lon})["amenity"="hospital"];
            out;
            """
            
            headers = {"User-Agent": self.user_agent}
            res = requests.get(overpass_url, params={"data": query}, headers=headers, timeout=90)
            res.raise_for_status()
            data = res.json()
            results = data.get("elements", [])
            
            if not results:
                print("⚠️ No hospitals found in OpenStreetMap within the specified radius.")
                return self._get_fallback_hospitals(place_name)
            
            print(f"✅ Found {len(results)} hospitals near {place_name}:\n")
            
            hospitals = []
            for el in results:
                tags = el.get("tags", {})
                name = tags.get("name", "Unnamed Hospital")
                
                # Address prioritization similar to your script
                address = tags.get("addr:full") or tags.get("addr:street") or tags.get("addr:city") or "Address not available"
                
                # Create OSM link
                osm_link = f"https://www.openstreetmap.org/node/{el['id']}"
                
                hospitals.append({
                    "name": name,
                    "address": address,
                    "contact_number": tags.get("phone") or tags.get("contact:phone") or "Contact not available",
                    "website": tags.get("website") or tags.get("contact:website") or osm_link
                })
            
            return hospitals if hospitals else self._get_fallback_hospitals(place_name)
                
        except Exception as e:
            print(f"❌ Error searching for hospitals: {e}")
            return self._get_fallback_hospitals(place_name)
    
    def _get_fallback_hospitals(self, place_name: str) -> List[Dict[str, str]]:
        """Return national helplines and crisis resources when no facilities found."""
        print(f"📞 Showing national helplines and crisis resources...")
        return [
            {
                "name": "KIRAN Mental Health Helpline (India)",
                "address": "Available nationwide - 24x7 Support",
                "contact_number": "1800-599-0019 (Toll-free)",
                "website": "https://kiran.nimhans.ac.in"
            },
            {
                "name": "Vandrevala Foundation Helpline",
                "address": "India - 24x7 Free Counseling",
                "contact_number": "9999 666 555",
                "website": "https://www.vandrevalafoundation.com"
            },
            {
                "name": "National Institute of Mental Health and Neurosciences (NIMHANS)",
                "address": "Hosur Road, Bengaluru, Karnataka 560029",
                "contact_number": "080-2699-5000",
                "website": "https://www.nimhans.ac.in"
            },
            {
                "name": "Suicide Prevention India Foundation",
                "address": "Available nationwide",
                "contact_number": "9820466726",
                "website": "http://www.spif.in"
            },
            {
                "name": "iCall Psychosocial Helpline",
                "address": "Available nationwide - Mon-Sat 10 AM - 8 PM",
                "contact_number": "9152987821",
                "website": "https://icallhelpline.org"
            }
        ]
    
    def get_crisis_resources(self) -> List[Dict[str, str]]:
        """Return a list of immediate crisis helplines and emergency resources."""
        return [
            {
                "name": "KIRAN Mental Health Helpline (India)",
                "contact_number": "1800-599-0019",
                "description": "24x7 toll-free mental health support line."
            },
            {
                "name": "Vandrevala Foundation Helpline",
                "contact_number": "9999 666 555",
                "description": "Free mental health counselling service across India."
            },
            {
                "name": "Suicide Prevention India Foundation",
                "contact_number": "9820466726",
                "description": "Suicide prevention and mental health support."
            },
            {
                "name": "Emergency Services (India)",
                "contact_number": "100",
                "description": "Police emergency helpline."
            },
            {
                "name": "Medical Emergency (India)",
                "contact_number": "108",
                "description": "Ambulance and medical emergency helpline."
            },
            {
                "name": "National Suicide Prevention Lifeline (US)",
                "contact_number": "988",
                "description": "24/7 confidential support for people in distress."
            }
        ]
    
    def add_emergency_contact(self, name: str, phone: str) -> bool:
        """
        Add a new emergency contact to the JSON file.
        
        Args:
            name: Contact name
            phone: Contact phone number
            
        Returns:
            True if contact was added successfully, False otherwise
        """
        try:
            import json
            import os
            from config import EMERGENCY_CONTACTS_JSON
            
            # Normalize phone number
            if not phone.startswith('+'):
                phone = '+91' + phone
            
            # Load existing contacts
            contacts = []
            if os.path.exists(EMERGENCY_CONTACTS_JSON):
                with open(EMERGENCY_CONTACTS_JSON, 'r') as f:
                    contacts = json.load(f)
            
            # Add new contact
            new_contact = {
                "name": name,
                "phone": phone
            }
            contacts.append(new_contact)
            
            # Save updated contacts
            with open(EMERGENCY_CONTACTS_JSON, 'w') as f:
                json.dump(contacts, f, indent=2)
            
            print(f"✅ Emergency contact added: {name} ({phone})")
            return True
            
        except Exception as e:
            print(f"❌ Error adding emergency contact: {e}")
            return False

    def get_emergency_contacts(self) -> list:
        """
        Get all emergency contacts from the JSON file.
        
        Returns:
            List of emergency contacts
        """
        try:
            import json
            import os
            from config import EMERGENCY_CONTACTS_JSON
            
            # Check if contacts file exists
            if not os.path.exists(EMERGENCY_CONTACTS_JSON):
                return []
            
            # Load emergency contacts
            with open(EMERGENCY_CONTACTS_JSON, 'r') as f:
                contacts = json.load(f)
            
            return contacts if isinstance(contacts, list) else []
            
        except Exception as e:
            print(f"❌ Error loading emergency contacts: {e}")
            return []

    def has_emergency_contacts(self) -> bool:
        """
        Check if any emergency contacts exist.
        
        Returns:
            True if emergency contacts exist, False otherwise
        """
        try:
            contacts = self.get_emergency_contacts()
            return len(contacts) > 0
        except Exception as e:
            print(f"❌ Error checking emergency contacts: {e}")
            return False


# ---------------------------------------------------------------------------
# Global instance
# ---------------------------------------------------------------------------
mental_health_service = MentalHealthService()
