"""Offline native controls use only fresh owned AppContainer names and temp files."""
import os
from uuid import uuid4

import pytest

from compute_plane.windows_appcontainer import AppContainerProfile


@pytest.mark.skipif(os.name != 'nt', reason='Windows native profile qualification')
def test_fresh_profile_is_created_and_closed_without_adopting_existing_identity():
    name = 'tare.tcp04.' + uuid4().hex
    with AppContainerProfile(name) as profile:
        assert profile.name == name and profile.sid_text.startswith('S-1-15-2-')
        assert profile.sid.value
        with pytest.raises(OSError):
            AppContainerProfile(name)
    profile.close()
    with pytest.raises(ValueError, match='closed'):
        profile.sid


@pytest.mark.skipif(os.name != 'nt', reason='Windows native profile qualification')
@pytest.mark.parametrize('name', ['', 'other.product.name', 'tare.tcp04.bad', 'tare.tcp04.' + 'f'*33])
def test_profile_name_is_validated_before_registration(name):
    with pytest.raises(ValueError):
        AppContainerProfile(name)
