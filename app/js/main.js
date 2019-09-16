
function openFileDialog(){
  var fileDlg = document.querySelector('#fileDlg');
  fileDlg.showModal();
}
function closeFileDialog(){
  var fileDlg = document.querySelector('#fileDlg');
  fileDlg.close();
}
function openProcessDialog(){
  var processDlg = document.querySelector('#processDlg');
  processDlg.showModal();
}
function closeProcessDialog(){
  var processDlg = document.querySelector('#processDlg');
  processDlg.close();
}
function openSearchDialog(){
  var searchDlg = document.querySelector('#searchDlg');
  searchDlg.showModal();
}
function closeSearchDialog(){
  var searchDlg = document.querySelector('#searchDlg');
  searchDlg.close();
}
function showResultsView(results){
  var path = require("path");
  searchResults = results;
  for(var i = 0; i < 24; i++){
    console.log(latentDataPath);
    if(results[i]["filename"].indexOf("ROLL") > -1){
      fileNames.push(path.basename(results[i]["filename"].replace(".dat", ".png")));
    }else{
      fileNames.push(path.basename(results[i]["filename"].replace(".dat", ".bmp")));
    }
    console.log(parseFloat(results[i]["score"]).toString());
    scores.push(parseFloat(results[i]["score"]).toString());
  }
  for(var i=0; i < 6; i++){
    resultImgs[i].attr("src", galleryDirPath + fileNames[i]);
    resultRanks[i].innerHTML = "Rank: " + (i+1).toString();
    resultScores[i].innerHTML = "Score: " + scores[i];
    // rgb(232, 0, 0) red
    // rgb(24, 214, 0) green
    // diff: 208, -214, 0
    if(parseFloat(scores[i]) >= 30){
      resultScores[i].style['color'] = 'rgba(24, 214, 0, 1)';
    }else{
      let fraction = 1.0 - parseFloat(scores[i])/30.0;
      let r = Math.round(24 + (fraction*208));
      let g = Math.round(214 + (fraction*-214));
      let rgb = `rgba(${r}, ${g}, 0, 1)`;
      resultScores[i].style['color'] = rgb;
    }
  }
  $("#resultsView").show();
}
function openDetailDialog(){
  // $("#detailImgSpan").zoom();
  var detailDlg = document.querySelector('#detailDlg');
  detailDlg.showModal();
  $("#detailModal").show();
}
function closeDetailDialog(){
  var detailDlg = document.querySelector('#detailDlg');
  detailDlg.close();
  $("#detailModal").hide();
}
function openCorrDialog(ind, numCorr){
  let rolledBaseName = path.basename(fileNames[ind], path.extname(fileNames[ind]))
  corrImgPath = scoreDirPath + "corr/" + latentBaseName + "_" + rolledBaseName + "_2.jpg";
  $("#corrImg").attr("src", corrImgPath);
  document.getElementById("corrScoreText").innerHTML = "Score: " + scores[ind];
  document.getElementById("corrNumText").innerHTML = "# of Corresponding Minutiae: " + numCorr;
  document.getElementById("corrFileText").innerHTML = "Filename: " + fileNames[ind];
  // $("#corrModal").trigger('zoom.destroy');
  // $("#corrModal").zoom({
  //   url: "../python/Data/current_latent_data/" + corrImgPath
  // });
  var corrDlg = document.querySelector('#corrDlg');
  corrDlg.showModal();
}
function closeCorrDialog(){
  var corrDlg = document.querySelector('#corrDlg');
  corrDlg.close();
}
function getFeatureImages(latentImgFileName, latentDataFileName){
  socket.emit('getFeatureImages', {"img":latentImgFileName, "dat":latentDataFileName});
}
function getSearchResults(latentDataFileName){
  socket.emit('getSearchResults', {"dat":latentDataFileName});
}
function getCorrImages(ind){
  let rolledImgFileName = fileNames[ind];
  socket.emit('getCorrImages', {"limg":latentImgPath, "rimg":galleryDirPath+rolledImgFileName, "ind": ind});
}

// Main
$("#resultsView").hide();
var latentImgPath = '';
var latentDataPath = '';
var latentBaseName = '';
var galleryDirPath = "/home/cori/research/LatentAFISV2/Matching_20181204/data/images/MSP/MSP_10K_images/";
var scoreDirPath = "/home/cori/research/LatentAFISV2/Matching_20181204/scores/";
var searchResults = {};
var pageNum = 0;
const MAX_PAGE_NUM = 3;
var detailImg = document.getElementById("latentImgDetail");
var corrImg = document.getElementById("corrImg");
const Chartist = require('chartist');
const path = require('path');

var resultImgs = [$("#resultImg1"), $("#resultImg2"), $("#resultImg3"),
                  $("#resultImg4"), $("#resultImg5"), $("#resultImg6")];
var resultRanks = [document.querySelector("#rankText1"), document.querySelector("#rankText2"),
                    document.querySelector("#rankText3"), document.querySelector("#rankText4"),
                    document.querySelector("#rankText5"), document.querySelector("#rankText6")];
var resultScores = [document.querySelector("#scoreText1"), document.querySelector("#scoreText2"),
                    document.querySelector("#scoreText3"), document.querySelector("#scoreText4"),
                    document.querySelector("#scoreText5"), document.querySelector("#scoreText6")];
var fileNames = [];
var scores = [];
var corr = [];

var socket = io.connect('http://localhost:8080');
socket.on('connect', function(msg){
});
socket.on('processing_complete', function(msg){
  closeProcessDialog();
  openDetailDialog();
  $("#closeBtn").hide();
  $("#searchBtn").show();
});
socket.on('search_complete', function(obj){
  closeSearchDialog();
  showResultsView(obj);
});
socket.on('corr_complete', function(msg){
  var ind = msg.ind;
  var numCorr = msg.numCorr;
  openCorrDialog(ind, numCorr);
});



$("#filePickBtn").click(function() {
  $("#latentImgDetail").attr("src", latentImgPath);
  $("#viewDetailBtn").attr("src", latentImgPath);
  wheelzoom(document.querySelectorAll('img'));
  document.getElementById("detailFileName").innerHTML = latentImgPath;
  getFeatureImages(latentImgPath, latentDataPath);
  closeFileDialog();
  openProcessDialog();
  return false;
});
$('input[type=file]').change(function(e){
  $in=$(this);
  $in.next().html($in.val());
  latentImgPath = document.getElementById("inputFile").files[0].path;
  latentDataPath = latentImgPath.replace(".bmp", ".dat").replace(".png", ".dat").replace("images", "templates");
  latentBaseName = path.basename(latentImgPath, path.extname(latentImgPath))
});
$("#searchBtn").click(function(){
  getSearchResults(latentDataPath);
  closeDetailDialog();
  openSearchDialog();
  $("#latentImgDetail").attr("src", latentImgPath);
  return false;
});
$("#closeBtn").click(function(){
  closeDetailDialog();
  $("#latentImgDetail").attr("src", latentImgPath);
  return false;
});
$("#xBtn").click(function(){
  closeCorrDialog();
  return false;
});
$("#viewDetailBtn").click(function(){
  openDetailDialog();
  $("#closeBtn").show();
  $("#searchBtn").hide();
  return false;
});

// buttons for viewing features
$("#originalBtn").click(function(){
  // $("#detailImgSpan").trigger('zoom.destroy');
    var newImg = new Image();
    newImg.src = latentImgPath;
    $("#latentImgDetail").attr("src", newImg.src);
  // $("#detailImgSpan").zoom({url: "../python/Data/Latent/" + latentImgFileName});
});
$("#OFBtn").click(function(){
  // $("#detailImgSpan").trigger('zoom.destroy');
    var newImg = new Image();

    newImg.src = latentImgPath.replace(".bmp", "_OF.jpeg").replace(".png", "_OF.jpeg").replace("latent", "latent/processed");
    $("#latentImgDetail").attr("src", newImg.src);

  // $("#detailImgSpan").zoom({
    // url: "../python/Data/current_latent_data/" + latentImgFileName.replace(".bmp", "_ROI.jpg"),
    // magnify: 0.5
  // });
});
$("#minutiae1Btn").click(function(){
  // $("#detailImgSpan").trigger('zoom.destroy');
    var newImg = new Image();
    newImg.src = latentImgPath.replace(".bmp", "_STFT_img.jpeg").replace(".png", "_STFT_img.jpeg").replace("latent", "latent/processed");
    $("#latentImgDetail").attr("src", newImg.src);
    document.querySelector("#latentImgDetail").style['background-size'] = 'cover';
  // $("#detailImgSpan").zoom({
    // url: "../python/Data/current_latent_data/" + latentImgFileName.replace(".bmp", "_OF.jpg"),
    // magnify: 0.5
  // });
});
$("#minutiae2Btn").click(function(){
  // $("#detailImgSpan").trigger('zoom.destroy');
    var newImg = new Image();
    newImg.src = latentImgPath.replace(".bmp", "_AEC_img.jpeg").replace(".png", "_AEC_img.jpeg").replace("latent", "latent/processed");
    $("#latentImgDetail").attr("src", newImg.src);
    document.querySelector("#latentImgDetail").style['background-size'] = 'cover';
  // $("#detailImgSpan").zoom({
    // url: "../python/Data/current_latent_data/" + latentImgFileName.replace(".bmp", "_minu1.jpg"),
    // magnify: 0.6
  // });
});
$("#minutiae3Btn").click(function(){
  // $("#detailImgSpan").trigger('zoom.destroy');
    var newImg = new Image();
    newImg.src = latentImgPath.replace(".bmp", "_common_2.jpeg").replace(".png", "_common_2.jpeg").replace("latent", "latent/processed");
    $("#latentImgDetail").attr("src", newImg.src);
    document.querySelector("#latentImgDetail").style['background-size'] = 'cover';
  // $("#detailImgSpan").delay().zoom({
    // url: "../python/Data/current_latent_data/" + latentImgFileName.replace(".bmp", "_minu2.jpg"),
    // magnify: 0.6
  // });
});

$("#upBtn").click(function(){
  console.log(pageNum);
  if(pageNum > 0){
    pageNum--;
    console.log("Turning back a page...");
    for(var i=0; i < 6; i++){
      var newImg = new Image();
      newImg.src = galleryDirPath + fileNames[i + (pageNum*6)];
      resultImgs[i].attr("src", newImg.src);
      resultRanks[i].innerHTML = "Rank: " + (i + (pageNum*6) + 1).toString();
      resultScores[i].innerHTML = "Score: " + scores[i + (pageNum*6)];
      // rgb(232, 0, 0) red
      // rgb(24, 214, 0) green
      // diff: 208, -214, 0
      if(parseFloat(scores[i + (pageNum*6)]) >= 30){
        resultScores[i + (pageNum*6)].style['color'] = 'rgba(24, 214, 0, 1)';
      }else{
        let fraction = 1.0 - parseFloat(scores[i + (pageNum*6)])/30.0;
        let r = Math.round(24 + (fraction*208));
        let g = Math.round(214 + (fraction*-214));
        let rgb = `rgba(${r}, ${g}, 0, 1)`;
        resultScores[i].style['color'] = rgb;
      }
    }
  }
});
$("#downBtn").click(function(){
  console.log(pageNum);
  if(pageNum < MAX_PAGE_NUM){
    pageNum++;
    console.log("Turning a page...");
    for(var i=0; i < 6; i++){
      var newImg = new Image();
      newImg.src = galleryDirPath + fileNames[i + (pageNum*6)];
      resultImgs[i].attr("src", newImg.src);
      resultRanks[i].innerHTML = "Rank: " + (i + (pageNum*6) + 1).toString();
      resultScores[i].innerHTML = "Score: " + scores[i + (pageNum*6)];
      // rgb(232, 0, 0) red
      // rgb(24, 214, 0) green
      // diff: 208, -214, 0
      if(parseFloat(scores[i + (pageNum*6)]) >= 30){
        resultScores[i + (pageNum*6)].style['color'] = 'rgba(24, 214, 0, 1)';
      }else{
        let fraction = 1.0 - parseFloat(scores[i + (pageNum*6)])/30.0;
        let r = Math.round(24 + (fraction*208));
        let g = Math.round(214 + (fraction*-214));
        let rgb = `rgba(${r}, ${g}, 0, 1)`;
        resultScores[i].style['color'] = rgb;
      }
    }
  }
});

// buttons for viewing candidate Correspondence
$("#resultImg1").click(function(){
  if(pageNum == 0){
    getCorrImages(0);
  }
});
$("#resultImg2").click(function(){
  if(pageNum == 0){
    getCorrImages(1);
  }
});
$("#resultImg3").click(function(){
  if(pageNum == 0){
    getCorrImages(2);
  }
});
$("#resultImg4").click(function(){
  if(pageNum == 0){
    getCorrImages(3);
  }
});
$("#resultImg5").click(function(){
  if(pageNum == 0){
    getCorrImages(4);
  }
});
$("#resultImg6").click(function(){
  if(pageNum == 0){
    getCorrImages(5);
  }
});

$("#corrBtn").click(function(){
  $("#corrImg").attr("src", corrImgPath);
});
$("#overlayBtn").click(function(){
  $("#corrImg").attr("src", "../python/Data/overlay_img.png");
});
