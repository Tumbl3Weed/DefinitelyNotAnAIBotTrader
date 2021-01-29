var convnetjs = require('convnetjs');
var math = require('mathjs');


var log = require('../core/log.js');

var config = require('../core/util.js').getConfig();

var SMMA = require('./indicators/SMMA.js');

/*

  BB strategy - okibcn 2018-01-03

 */
// helpers
var _ = require('lodash');
var log = require('../core/log.js');

// var BB = require('./indicators/BB.js');
var rsi = require('./indicators/RSI.js');

var neuralnetcal = {
  // stores the candles
  priceBuffer: [],
  priceVolumeBuffer: [],
  priceBufferSMA: [],
  priceBufferHigh: [],
  priceBufferLow: [],
  priceBufferOpen: [],
  priceBufferClose: [],
  adxrBuffer: [],
  predictionCount: 0,

  batchsize: 1,
  // no of neurons for the layer
  layer_neurons: 0,
  // activaction function for the first layer, when neurons are > 0
  layer_activation: 'tahn',
  // normalization factor
  scale: 1,
  // stores the last action (buy or sell)
  prevAction: 'wait',
  //stores the price of the last trade (buy/sell)
  prevPrice: 0,
  trailPriceBuy: 0,
  trailPriceSell: 0,
  // counts the number of triggered stoploss events
  // stoplossCounter: 0,
  actionIncentive: 1,
  // if you want the bot to hodl instead of selling during a small dip
  // use the hodl_threshold. e.g. 0.95 means the bot won't sell
  // when unless the price drops below a 5% threshold of the last buy price (this.privPrice)
  hodl_threshold: 0.96,

  // init the strategy
  init: function () {

    this.name = 'neuralnetcal';
    this.requiredHistory = config.tradingAdvisor.historySize;
    this.lastPredictionLossPercent = 100;
    this.bestLossPercent = 100;
    this.lastPrediction = 1;
    this.predictionMinPercent = this.settings.minPredictionAccuracy;
    // smooth the input to reduce the noise of the incoming data
    this.SMMA = new SMMA(3);


    let layers = [
      { type: 'input', out_sx: 1, out_sy: 1, out_depth: 7 },
      { type: 'fc', num_neurons: this.layer_neurons },
      { type: 'fc', num_neurons: this.layer_neurons, activation: this.layer_activation },
      { type: 'fc', num_neurons: this.layer_neurons, activation: this.layer_activation },
      { type: 'fc', num_neurons: this.layer_neurons },
      { type: 'fc', num_neurons: this.layer_neurons, activation: this.layer_activation },
      { type: 'fc', num_neurons: this.layer_neurons, activation: this.layer_activation },
      { type: 'regression', num_neurons: 1 }
    ];

    this.nn = new convnetjs.Net();

    this.nn.makeLayers(layers);
    this.trainer = new convnetjs.SGDTrainer(this.nn, {
      method: 'adadelta',
      learning_rate: this.settings.learning_rate,
      momentum: this.settings.momentum,
      batch_size: this.batchsize,
      l2_decay: this.settings.decay
    });

    this.requiredHistory = this.tradingAdvisor.historySize;

    console.log(this.settings.bbands);
    this.addIndicator('bb', 'BB', this.settings.bbands);
    this.addIndicator('rsi', 'RSI', this.settings);

    var adxrsettings = {
      optInTimePeriod: 10
    };
    this.addTulipIndicator('adxr', 'adxr', adxrsettings);
    this.hodl_threshold = this.settings.hodl_threshold;
  },

  setNormalizeFactor: function (candle) {
    this.scale = Math.pow(10, Math.trunc(candle.vwp).toString().length + 2);
    //log.debug('Set normalization factor to',this.scale);
  },

  learn: function (price) {

    for (let i = 1; i < this.priceBufferSMA.length; i++) {
      let current_price = [this.priceBuffer[i]];
      //let testVal = Math.random();
      let testVal = this.priceBufferSMA[i - 1];
      let testVal2 = this.priceBuffer[i - 1];
      let testVal3 = this.priceBufferHigh[i - 1];
      let testVal4 = this.priceBufferLow[i - 1];
      let testVal5 = this.priceBufferOpen[i - 1];
      let testVal6 = this.priceBufferClose[i - 1];
      let testVal7 = this.priceVolumeBuffer[i - 1];
      let testVal8 = this.adxrBuffer[i - 1];
      let vol = new convnetjs.Vol([testVal2, testVal, testVal3, testVal4, testVal5, testVal6, testVal7, testVal8]);
      this.trainer.train(vol, current_price);
      this.nn.forward(vol, true);
      this.predictionCount++;
    }

    for (let index = 0; index < 4; index++) {
      for (let i = Math.round(this.priceBufferSMA.length * 0.80); i < this.priceBufferSMA.length; i++) {
        let current_price = [this.priceBuffer[i]];
        //let testVal = Math.random();
        let testVal = this.priceBufferSMA[i - 1];
        let testVal2 = this.priceBuffer[i - 1];
        let testVal3 = this.priceBufferHigh[i - 1];
        let testVal4 = this.priceBufferLow[i - 1];
        let testVal5 = this.priceBufferOpen[i - 1];
        let testVal6 = this.priceBufferClose[i - 1];
        let testVal7 = this.priceVolumeBuffer[i - 1];
        let testVal8 = this.adxrBuffer[i - 1];
        let vol = new convnetjs.Vol([testVal2, testVal, testVal3, testVal4, testVal5, testVal6, testVal7, testVal8]);
        this.trainer.train(vol, current_price);
        this.nn.forward(vol, true);
        this.predictionCount++;
      }
    }

    for (let x = 0; x < 10; x++) {
      let testVal = this.priceBufferSMA[this.priceBufferSMA.length - 2];
      let testVal2 = this.priceBuffer[this.priceBufferSMA.length - 2];
      let testVal3 = this.priceBufferHigh[this.priceBufferSMA.length - 2];
      let testVal4 = this.priceBufferLow[this.priceBufferSMA.length - 2];
      let testVal5 = this.priceBufferOpen[this.priceBufferSMA.length - 2];
      let testVal6 = this.priceBufferClose[this.priceBufferSMA.length - 2];
      let testVal7 = this.priceVolumeBuffer[this.priceBufferSMA.length - 2];
      let testVal8 = this.adxrBuffer[this.priceBufferSMA.length - 2];
      let vol = new convnetjs.Vol([testVal2, testVal, testVal3, testVal4, testVal5, testVal6, testVal7, testVal8]);
      this.trainer.train(vol, price);
      this.nn.forward(vol, true);
      this.predictionCount++;

    }


  },

  predictCandle: function () {
    let indexVal = this.priceBuffer.length - 1;
    let testVal = this.priceBufferSMA[indexVal];
    let testVal2 = this.priceBuffer[indexVal];
    let testVal3 = this.priceBufferHigh[indexVal];
    let testVal4 = this.priceBufferLow[indexVal];
    let testVal5 = this.priceBufferOpen[indexVal];
    let testVal6 = this.priceBufferClose[indexVal];
    let testVal7 = this.priceVolumeBuffer[indexVal];
    let testVal8 = this.adxrBuffer[indexVal];
    let vol = new convnetjs.Vol([testVal2, testVal, testVal3, testVal4, testVal5, testVal6, testVal7, testVal8]);//

    let prediction = this.nn.forward(vol);
    return prediction.w[0];
  },

  predictPrevCandle: function () {
    let indexVal = this.priceBuffer.length - 2;
    let testVal = this.priceBufferSMA[indexVal];
    let testVal2 = this.priceBuffer[indexVal];
    let testVal3 = this.priceBufferHigh[indexVal];
    let testVal4 = this.priceBufferLow[indexVal];
    let testVal5 = this.priceBufferOpen[indexVal];
    let testVal6 = this.priceBufferClose[indexVal];
    let testVal7 = this.priceVolumeBuffer[indexVal];
    let testVal8 = this.adxrBuffer[indexVal];
    let vol = new convnetjs.Vol([testVal2, testVal, testVal3, testVal4, testVal5, testVal6, testVal7, testVal8]);//

    let prediction = this.nn.forward(vol);
    return prediction.w[0];
  },

  update: function (candle) {
    let price = candle.vwp;
    if (price < this.trailPriceBuy) {
      //this.trailPriceBuy = Math.round(price * 0.9975);
      this.trailPriceBuy = price;
    }
    if (price > this.trailPriceSell) {
      //this.trailPriceSell = price * 1.0025;
      this.trailPriceSell = price;
    }
    var adxrVal = this.tulipIndicators.adxr.result.result;

    // play with the candle values to finetune this
    this.SMMA.update(price);
    let smmaFast = this.SMMA.result;
    if (1 === this.scale && 1 < candle.high && 0 === this.predictionCount) this.setNormalizeFactor(candle);

    if (smmaFast / this.scale != 0) {
      this.priceBufferSMA.push(smmaFast / this.scale);
      this.priceBuffer.push(price / this.scale);
      this.priceVolumeBuffer.push(candle.volume / 100);
      this.priceBufferLow.push(candle.low / this.scale);
      this.priceBufferHigh.push(candle.high / this.scale);
      this.priceBufferOpen.push(candle.open / this.scale);
      this.priceBufferClose.push(candle.close / this.scale);
      this.adxrBuffer.push(adxrVal / 100);
    }

    if (10 > this.priceBufferSMA.length) return;

    if (this.adxrBuffer.length > this.settings.price_buffer_len) {
      this.priceBufferSMA.splice(0, 1);
      this.priceBuffer.splice(0, 1);
      this.priceVolumeBuffer.splice(0, 1);
      this.priceBufferLow.splice(0, 1);
      this.priceBufferHigh.splice(0, 1);
      this.priceBufferOpen.splice(0, 1);
      this.priceBufferClose.splice(0, 1);
      this.adxrBuffer.splice(0, 1);
    }

    for (tweakme = 0; tweakme < 20; ++tweakme)
      this.learn(candle.vwp);
  },

  processTrade: function (event) {
    console.log('alter completed trade events');
    if (event.cancel) {
      console.log('dont alter completed trade events');
      return;
    }
    this.prevPrice = event.effectivePrice;
    this.prevAction = event.action;
    // store the price of the previous trade

  },

  check: function (candle) {

    let price = candle.vwp;

    if (this.predictionCount < this.settings.min_predictions || this.priceBuffer.length < this.requiredHistory) {
      this.prevPrice = price;
      this.trailPriceBuy = price;
      this.trailPriceSell = price;
      return;
    }
    var usingNueral = false;

    if (this.lastPredictionLossPercent < this.predictionMinPercent * this.actionIncentive) { //only if we are accurate enough use nueral logic and take action
      usingNueral = true;
      this.actionIncentive *= 0.98;
      for (tweakme = 0; tweakme < 20; ++tweakme)
        this.learn(candle.vwp);
    } else {
      // this.actionIncentive *= 0.9999;
      while (Math.abs(100 - ((price / this.predictPrevCandle) * 100)) < this.predictionMinPercent * this.actionIncentive) {
        for (tweakme = 0; tweakme < 200; ++tweakme)
          this.learn(candle.vwp);
      }
      usingNueral = true;
    }
    // if (!usingNueral) {
    //   for (tweakme = 0; tweakme < 200; ++tweakme)
    //     this.learn(candle.vwp);
    // } else {
    //   for (tweakme = 0; tweakme < 20; ++tweakme)
    //     this.learn(candle.vwp);
    // }

    this.lastPredictionLossPercent = Math.abs(100 - ((price / this.lastPrediction) * 100));
    if (this.bestLossPercent * 0.99 > this.lastPredictionLossPercent) {
      this.bestLossPercent = this.lastPredictionLossPercent;
      console.log('!!!!!!!!!!!!!!best prediction so far = ' + this.bestLossPercent + '!!!!!!!!!!!!!!!');
    }
    //log.info(this.lastPredictionLossPercent);

    let prediction = this.predictCandle() * this.scale;
    // console.log('prediction ' + prediction);
    this.lastPrediction = prediction;
    let meanp = math.mean(prediction, price);
    let alpha = ((prediction / price) - 1) * 100;
    let meanAlpha = ((meanp - price) / price) * 100;

    let signalPredictisDown = prediction < price; //false means buy true means sell, rising price prediction.

    // console.log(this.priceBuffer[this.priceBuffer.length - 2] / this.priceBuffer[this.priceBuffer.length - 1], this.priceBuffer[this.priceBuffer.length - 2] / this.priceBuffer[this.priceBuffer.length - 1] < 1,
    //   (!signalPredictisDown), prediction / price > 1);

    if (usingNueral)

      if ((!signalPredictisDown) && prediction / price > 1 && this.priceBuffer[this.priceBuffer.length - 2] / this.priceBuffer[this.priceBuffer.length - 1] < 1) {// && this.trailPriceBuy < prediction
        console.log('using nueral =' + usingNueral);

        console.log("Buy - Predicted variation: " + meanAlpha);
        log.debug('nueral buy ' + this.lastPredictionLossPercent);
        if (price < this.prevPrice * 0.9975) {
          return this.advice('long', price, true); //instant buy(takeProfit if available)
        } else {
          return this.advice('long', price, false);
        }


      }
    if (usingNueral)

      if (signalPredictisDown && (prediction / price) < 1 && this.priceBuffer[this.priceBuffer.length - 2] / this.priceBuffer[this.priceBuffer.length - 1] > 1) {//&& this.prevPrice * 1 < price) {// && this.prevPrice * 0.97 < prediction ) { //this.trailPriceSell > prediction && 
        console.log('using nueral =' + usingNueral);

        console.log("Sell - Predicted variation: " + meanAlpha);
        log.debug('nueral sell ' + this.lastPredictionLossPercent);
        if (price > this.prevPrice * 1.0025) {
          return this.advice('short', price, true); //instant sell  (takeProfit if available)
        } else {
          return this.advice('short', price, false);
        }
      }

  },

  end: function () {
    //log.debug('Triggered stoploss',this.stoplossCounter,'times');
    console.log('Triggered stoploss' + this.stoplossCounter + 'times');
  }
};

module.exports = neuralnetcal;
