package org.sbml.spatial.segmentation;

import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.learning.config.Adam;
import org.deeplearning4j.zoo.model.UNet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.File;
import java.io.IOException;
import java.util.Random;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.util.HashMap;
import java.util.Map;


//This class is for training the UNet model on any number of images and then saving it.
public class UnetTrainAndSave {

	    private static final Logger log = LoggerFactory.getLogger(UnetTrainAndSave.class);
	    private static final int WIDTH = 128;
	    private static final int HEIGHT = 128;
	    private static final int CHANNELS = 3;
	    
	  //This performs in-line replacement of RGB Type to BGR type for better memory performance
	    public static BufferedImage getBGRBufferedImage(BufferedImage bufferedImage) {
	        for (int w = 0; w < bufferedImage.getWidth(); w++) {
	            for (int h = 0; h < bufferedImage.getHeight(); h++) {
	                int p = bufferedImage.getRGB(w, h);
	                int a = (p >> 24) & 0xff;
	                int r = (p >> 16) & 0xff;
	                int g = (p >> 8) & 0xff;
	                int b = p & 0xff;
	                // swap r (red) and b (blue) channels
	                p = (a << 24) | (b << 16) | (g << 8) | r;
	                bufferedImage.setRGB(w, h, p);
	            }
	        }
	        return bufferedImage;
	    }

	    public static BufferedImage getBGRBufferedImage(File imageFile) {
	        try {
	            BufferedImage bufferedImage = ImageIO.read(imageFile);
	            return getBGRBufferedImage(bufferedImage);
	        } catch (IOException e) {
	            e.printStackTrace();
	        }
	        return null;
	    }

	    public static void main(String[] args) {
	        try {
	            int batchSize = 10;
	            //String home = System.getProperty("user.home");
	            String directory = System.getProperty("user.dir");
	            //String dataPath = directory + File.separator + "dataset";

	            DataNormalization scaler = new ImagePreProcessingScaler(); // scale image between 0 and 1
	            UnetPathLabelGenerator labeler = new UnetPathLabelGenerator();

	            File rootDir = new File("C:\\Users\\Subroto\\Desktop\\validation\\44_set\\train1");
	            //File rootDir = new File(dataPath + File.separator + "small_dataset");
	            String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
	            Random rng = new Random();
	            FileSplit inputSplit = new FileSplit(rootDir,allowedExtensions,rng);
	            ImageRecordReader imageRecordReader = new ImageRecordReader(HEIGHT,WIDTH,CHANNELS,labeler);
	            imageRecordReader.initialize(inputSplit);
	            int labelIndex = 1;
	            DataSetIterator imageDataSetIterator = new RecordReaderDataSetIterator(imageRecordReader,batchSize,labelIndex,labelIndex,true);
	            scaler.fit(imageDataSetIterator);
	            imageDataSetIterator.setPreProcessor(scaler);


	            Map<Integer, Double> learningScheduleMap = new HashMap<>();
	            learningScheduleMap.put(0, 0.00005);
	            learningScheduleMap.put(200, 0.00001);
	            learningScheduleMap.put(600, 0.000005);
	            learningScheduleMap.put(800, 0.0000001);
//	            learningScheduleMap.put(1000, 0.00001);

	            int numEpochs = 100;
	           
	            ComputationGraph model  = UNet.builder().updater(new Adam(1e-4)).build().init();
//	            ComputationGraph model  = UNet.builder().updater(new Adam(new MapSchedule(ScheduleType.ITERATION, learningScheduleMap))).build().init();
	            model.addListeners(new ScoreIterationListener());
	            model.fit(imageDataSetIterator,numEpochs);

	            //log.warn(model.summary());
	            
	            log.info("*****SAVE MODEL******");
	            
	            //Location for saving the model
	            //File locationTosave = new File(home + File.separator + "unetSave.zip");
	            File locationTosave = new File(directory + File.separator + "saveModel" + File.separator + "unetSave.zip");
	            boolean saveUpdater = false;
	            //ModelSerializer needs Model name, Location of saving the model and saveUpdater. 
	            ModelSerializer.writeModel(model,locationTosave,saveUpdater);
	            
	        } catch (Exception e) {
	            System.err.println("Oooooops");
	            e.printStackTrace();
	        }
	    }
}

