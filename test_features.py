import unittest
import json
from app import app, db
from models import User, Favorite, Comment

class FeatureTestCase(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        self.client = app.test_client()
        with app.app_context():
            db.create_all()

    def tearDown(self):
        with app.app_context():
            db.session.remove()
            db.drop_all()

    def register(self, username, password):
        return self.client.post('/api/register', json={
            'username': username,
            'password': password
        })

    def login(self, username, password):
        return self.client.post('/api/login', json={
            'username': username,
            'password': password
        })

    def test_auth_workflow(self):
        # Test Register
        resp = self.register('testuser', 'password123')
        self.assertEqual(resp.status_code, 200)
        self.assertIn('注册成功', resp.get_json()['message'])

        # Test Login
        resp = self.login('testuser', 'password123')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()['user']['username'], 'testuser')

        # Test Me
        resp = self.client.get('/api/me')
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.get_json()['authenticated'])

    def test_favorites(self):
        self.register('favuser', 'pass')
        self.login('favuser', 'pass')

        video_data = {
            'id': '12345',
            'title': 'Test Video',
            'cover': 'http://example.com/cover.jpg'
        }

        # Add Favorite
        resp = self.client.post('/api/favorites/add', json=video_data)
        self.assertEqual(resp.status_code, 200)
        self.assertIn('收藏成功', resp.get_json()['message'])

        # Get Favorites
        resp = self.client.get('/api/favorites')
        favs = resp.get_json()['videos']
        self.assertEqual(len(favs), 1)
        self.assertEqual(favs[0]['id'], '12345')

        # Remove Favorite
        resp = self.client.post('/api/favorites/remove', json={'id': '12345'})
        self.assertEqual(resp.status_code, 200)
        self.assertIn('已取消收藏', resp.get_json()['message'])

    def test_comments(self):
        self.register('commenter', 'pass')
        self.login('commenter', 'pass')

        # Add Comment
        resp = self.client.post('/api/comments/add', json={
            'video_id': 'video1',
            'content': 'Great video!'
        })
        self.assertEqual(resp.status_code, 200)
        self.assertIn('评论成功', resp.get_json()['message'])

        # Get Comments
        resp = self.client.get('/api/comments?video_id=video1')
        comments = resp.get_json()['comments']
        self.assertEqual(len(comments), 1)
        self.assertEqual(comments[0]['content'], 'Great video!')
        self.assertEqual(comments[0]['username'], 'commenter')

    def test_membership_upgrade(self):
        self.register('member', 'pass')
        self.login('member', 'pass')

        # Check initial status
        resp = self.client.get('/api/me')
        self.assertFalse(resp.get_json()['is_member'])

        # Upgrade
        resp = self.client.post('/api/upgrade')
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.get_json()['is_member'])

        # Check updated status
        resp = self.client.get('/api/me')
        self.assertTrue(resp.get_json()['is_member'])

if __name__ == '__main__':
    unittest.main()
