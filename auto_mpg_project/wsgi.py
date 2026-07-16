"""WSGI config for the Auto MPG project."""
import os

from django.core.wsgi import get_wsgi_application


os.environ.setdefault("DJANGO_SETTINGS_MODULE", "auto_mpg_project.settings")

application = get_wsgi_application()
