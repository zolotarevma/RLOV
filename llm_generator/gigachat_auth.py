import requests
import uuid
from config import GIGACHAT_AUTHORIZATION_KEY


def get_access_token():
    """Получает временный токен доступа для GigaChat API."""
    if not GIGACHAT_AUTHORIZATION_KEY:
        raise ValueError("GIGACHAT_AUTHORIZATION_KEY не установлен.")

    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    payload = {'scope': 'GIGACHAT_API_PERS'}
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json',
        'RqUID': str(uuid.uuid4()),
        'Authorization': f'Basic {GIGACHAT_AUTHORIZATION_KEY}'
    }

    try:
        response = requests.post(url, headers=headers, data=payload, verify=False)
        response.raise_for_status()
        return response.json().get('access_token')
    except requests.exceptions.RequestException as e:
        print(f"[GigaChat Auth] Ошибка получения токена: {e}")
        return None
