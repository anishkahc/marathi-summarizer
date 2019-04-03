const summarizer = require('nodejs-text-summarizer')
const translate = require('@vitalets/google-translate-api');
var express = require('express');
var app = express();
var bodyParser = require('body-parser');
app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());
app.set('view engine', 'ejs');

app.post('/', function (req, response) {
translate(req.body.data, {client: 'gtx' ,to: 'en'}).then(res => {
    let result = summarizer(res.text)
    translate(result, {client: 'gtx' ,to: 'mr'}).then(res => {
	response.render('index',{ data : res.text});
    }).catch(err => {
        console.error(err);
    });     
}).catch(err => {
    console.error(err);
}); 
})



var server=app.listen(3000,function() {});