'use strict';

const app = require('electron').app;
const BrowserWindow = require('electron').BrowserWindow;
const ipc = require('electron').ipc;
//const server = require('http').createServer();
const io = require('socket.io').listen(8080);
const path = require('path');
const fs = require('fs');

var pwd = app.getAppPath();

io.sockets.on('connection', function(socket){
    console.log('Connection established.');
    socket.on('generateTemplates', function(msg){
        console.log('Generating latent templates...');
        console.log(msg);
        var latent_img_path = msg.img;
        var latent_data_path = msg.dat;
        let featureExtractionPath = '/home/cori/research/LatentAFISV2/Matching_20181204/system_2/extraction_Latent_12022018.py';
        console.log(featureExtractionPath);
        let extractionProcess = require('child_process').spawn('python', [featureExtractionPath, "--image_path="+latent_img_path, "--data_path="+latent_data_path]);
        if (extractionProcess != null) {
          console.log('Template Generation: Success')
        }
        extractionProcess.stdout.on('data', (data) => {
          console.log(`stdout: ${data}`);
        });
        extractionProcess.stderr.on('data', (data) => {
          console.log(`stderr: ${data}`);
        });
        extractionProcess.on('close', (code) => {
          io.emit('getFeatureImages');
        });

        io.emit('getFeatureImages');
    });
    socket.on('getFeatureImages', function(msg){
        console.log('Getting feature images...');
        console.log(msg);
        var latent_img_path = msg.img;
        var latent_data_path = msg.dat;
        let loadLatentPath = path.join(__dirname, 'python', 'load_latent.py');
        // const pyPath2 = path.join(__dirname, 'hello.py');
        console.log(loadLatentPath);
        let loadProcess = require('child_process').spawn('python', [loadLatentPath, latent_img_path, latent_data_path]);
        // const loadProcess = require('child_process').spawn('python', [pyPath2]);
        if (loadProcess != null) {
          console.log('loadProcess success')
        }
        loadProcess.stdout.on('data', (data) => {
          console.log(`stdout: ${data}`);
        });
        loadProcess.stderr.on('data', (data) => {
          console.log(`stderr: ${data}`);
        });
        loadProcess.on('close', (code) => {
          io.emit('processing_complete');
        });

        io.emit('processing_complete');
    });
    socket.on('getSearchResults', function(msg){
        console.log('Getting search results ...');
        console.log(msg);
        var latent_data_path = msg.dat
        var score_path = msg.dat.replace('.dat', '.csv');
        let runMatcherPath = pwd.replace("LatentAFISGUI-electron", "Matching_20181204/integrated_matcher/match");
        console.log(runMatcherPath);
        var execFile = require('child_process').execFile;
        //let searchProcess = require('child_process').spawn('python', [runMatcherPath, latent_data_path]);
        let searchProcess = execFile(runMatcherPath, [latent_data_path, score_path], function(error, stdout, stderr){

            console.log(error);
            console.log(stderr);
            console.log(stdout);
            var obj;
            const csv = require('csvtojson');
            csv()
                .fromFile(score_path)
                .then((jsonObj)=>{
                    io.emit('search_complete', jsonObj);
                })
        });
    });
    socket.on('getCorrImages', function(msg){
        console.log("Getting corr images...");
        console.log(msg);
        var latent_img_path = msg.limg;
        var rolled_img_path = msg.rimg;
        let numCorr = 0;
        var ind = msg.ind;
        let runCorrPath = path.join(__dirname, 'python', 'get_correspondence.py');
        let corrProcess = require('child_process').spawn('python',
            [runCorrPath, latent_img_path, rolled_img_path]);
        if(corrProcess != null){
            console.log('corrProcess success');
        }
        corrProcess.stdout.on('data', (data) =>{
            numCorr = parseInt(data);
            console.log('stdout: ', numCorr);
        });
        corrProcess.stderr.on('data', (data) =>{
            console.log(`stderr: ${data}`);
        });
        corrProcess.on('close', (code) => {
            io.emit('corr_complete', {"ind":ind, "numCorr":numCorr});
        });
    });

});

var mainWindow = null;

app.on('ready', function() {
    mainWindow = new BrowserWindow({
        height: 850,
        width: 1430,
        minHeight: 850,
        minWidth: 1430,
        icon: './favicon-32x32.png'
    });
    //mainWindow.setMenu(null);
    mainWindow.loadURL('file://' + __dirname + '/app/main.html');
});
