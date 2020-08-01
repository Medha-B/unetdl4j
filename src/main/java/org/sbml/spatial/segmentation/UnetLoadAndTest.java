package org.sbml.spatial.segmentation;


import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;

//This is for loading the saved UNet model and then testing it.
public class UnetLoadAndTest {

	    private static final Logger log = LoggerFactory.getLogger(UnetLoadAndTest.class);
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
	            
	        	//String home = System.getProperty("user.home");
	        	String directory = System.getProperty("user.dir");
	        	String dataPath = directory + File.separator + "dataset";
	        	
	            String pathToImage;
	            if (args.length > 0) {
	                pathToImage = args[0];
	            } else {
	            	pathToImage = dataPath + File.separator + "raw_images" + File.separator + "F01_621w1_crop13.tif";
	            }
	            
	            
                log.info("*****LOAD MODEL******");
	            //Location where the model is saved
                //File locationTosave = new File(home + File.separator + "unetSave.zip"); //Depends upon where the model weights are actually saved
                File locationTosave = new File(directory + File.separator + "saveModel" + File.separator + "unetSave.zip");
	            ComputationGraph model  = ModelSerializer.restoreComputationGraph(locationTosave); 

	            
	            log.info("*****EVALUATE MODEL******");
	            NativeImageLoader loader = new NativeImageLoader(HEIGHT, WIDTH, CHANNELS);
	            BufferedImage bufferedBGR = getBGRBufferedImage(new File(pathToImage));
	            INDArray imageNative = loader.asMatrix(bufferedBGR);

	            log.warn(imageNative.shapeInfoToString());

	            imageNative = imageNative.reshape(1, CHANNELS, HEIGHT, WIDTH);
	            imageNative = imageNative.divi(255f);

	            INDArray[] output = model.output(imageNative);
	            for (INDArray out : output) {
	                out = out.reshape(1, HEIGHT,WIDTH);
	                //out = out.permute(2,1,0);
	                BufferedImage bufferedImage = new BufferedImage(WIDTH,HEIGHT,BufferedImage.TYPE_BYTE_GRAY);
	                for (int i=0; i < WIDTH; i++) {
	                    for (int j=0; j < HEIGHT; j++) {
	                        float f = out.getFloat(new int[]{0,j,i});
	                        int gray = (int)(f*255.0);
	                        if (gray > 127) {
	                            bufferedImage.setRGB(i,j,Color.WHITE.getRGB());
	                        } else {
	                            bufferedImage.setRGB(i,j,Color.BLACK.getRGB());
	                        }
//	                      bufferedImage.setRGB(i,j,new Color(gray,gray,gray).getRGB());
	                    }
	                }
	                ImageIO.write(bufferedImage,"tif",new File(directory + File.separator + "Output" + File.separator + "outputUnet.tif"));
	                //ImageIO.write(bufferedImage,"tif",new File(home + File.separator + "outputUnet.tif"));
	            }
	        } catch (Exception e) {
	            System.err.println("Oooooops");
	            e.printStackTrace();
	        }
	    }
}
