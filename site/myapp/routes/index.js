var express = require('express');
var upload_files = require('multer')();
var router = express.Router();

let landing = require('../controllers/landing');
let socket_api = require("../socket_api");


function check_api_key(req, res, next) {
    const api_key = req.get("API-Key");
    if (!api_key || api_key !== process.env.FF_API_KEY) {
        res.status(401).json({error: "unauthorized"});
    }
    else {
        next();
    }
}


router.get('/', landing.get_sign_in);
router.post('/', landing.post_sign_in);

router.get('/logout', landing.logout);


router.get('/home/:username', landing.get_home);
router.post('/home/:username', landing.post_home);

router.get('/admin', landing.get_admin);
router.post('/admin', landing.post_admin);

router.post('/image_set_upload', upload_files.array('source_file[]'), landing.post_image_set_upload);
router.post('/upload_notification', check_api_key, socket_api.post_upload_notification);
router.post('/progress_notification', check_api_key, socket_api.post_progress_notification);

module.exports = router;
