import requests

class AuthManager:
    def __init__(self, email, password):
        self.identity_api_key = "AIzaSyC5x1Zhy8kwmkD-LbH8VTQoy0wOhmSpv_w"
        self.cloud_function_url = "https://weather-auth-handler-586857630076.europe-west1.run.app"
        
        # Authenticate with Identity Platform
        self.id_token = self._login(email, password)

    def _login(self, email, password):
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={self.identity_api_key}"
        payload = {"email": email, "password": password, "returnSecureToken": True}
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            return response.json()["idToken"]
        raise PermissionError("Login failed.")

    def request_access(self, service_name, details=None):
        """Asks the Cloud Function for keys to a specific service."""
        headers = {"Authorization": f"Bearer {self.id_token}"}
        payload = {
            "service": service_name,
            "details": details or {}
        }
        
        response = requests.post(self.cloud_function_url, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()
        raise PermissionError(f"Access denied for {service_name}.")