package cs276.pa4;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

public class PairwiseLearner extends Learner {
  private LibSVM model;
  public PairwiseLearner(boolean isLinearKernel){
    try{
      model = new LibSVM();
    } catch (Exception e){
      e.printStackTrace();
    }
    
    if(isLinearKernel){
      model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
    }
  }
  
  public PairwiseLearner(double C, double gamma, boolean isLinearKernel){
    try{
      model = new LibSVM();
    } catch (Exception e){
      e.printStackTrace();
    }
    
    model.setCost(C);
    model.setGamma(gamma); // only matter for RBF kernel
    if(isLinearKernel){
      model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
    }
  }
  
	@Override
	public Instances extract_train_features(String train_data_file,
			String train_rel_file, Map<String, Double> idfs) {
		
		Standardize filter = new Standardize();
		Instances dataset = null;
		
		/* Build attributes list */
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("body_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));
		attributes.add(new Attribute("relevance_score"));
		dataset = new Instances("train_dataset", attributes, 0);
		
		/* Add data */
		// query -> documents list
		Map<Query,List<Document>> trainData = null;
		// query -> (url -> score)
		Map<String, Map<String, Double>> trainRels = null;
		try{
			trainData = Util.loadTrainData(train_data_file);
			trainRels = Util.loadRelData(train_rel_file);
		}catch(Exception e){
			System.err.println("Error loading training data");
		}
		//Need to keep track of what instances belong to which query and also their index
		//in the standardized instances object
		Map<Query, List<Integer>> mapInstance = new HashMap<Query, List<Integer>>();
		int index = 0;
		for (Map.Entry<Query, List<Document>> entry : trainData.entrySet()){
			Query q = entry.getKey();
			List<Integer> list = new ArrayList<Integer>();
			List<Document> documents = entry.getValue();
			for(Document d : documents){
				double[] instance = FormatDocument.createInstanceVector(d, q, idfs, trainRels);
				Instance inst = new DenseInstance(1.0,instance);
				list.add(new Integer(index));
				dataset.add(inst);
				index = index + 1;
			}
			mapInstance.put(q, list);
		}
		/* Set last attribute as target */
		dataset.setClassIndex(dataset.numAttributes() - 1);
		Instances new_instances = null;
		// Standardize all of the instances
		try{
			filter.setInputFormat(dataset);
			new_instances = Filter.useFilter(dataset, filter);
		}catch(Exception e){
			System.err.println("Error filtering dataset");
		}
		// Take the difference of all the documents that belong to a query and add to difference_instances
		Instances difference_instances = new Instances("train_dataset", attributes, 0);
		for(Map.Entry<Query, List<Integer>> entry : mapInstance.entrySet()){
			List<Integer> list = entry.getValue();
			for(Integer i : list){
				for(Integer j : list){
					Instance a = new_instances.get(i.intValue());
					Instance b = new_instances.get(j.intValue());
					Instance diff = getNewInstance(a,b);
					if(diff != null){
						difference_instances.add(diff);
					}
				}
			}
		}
		return difference_instances;
	}
	
	private Instance getNewInstance(Instance aInst, Instance bInst){
		double[] a = aInst.toDoubleArray();
		double[] b = bInst.toDoubleArray();
		double[] diff = new double[a.length];
		for(int i = 0; i < a.length; i++){
			diff[i] = a[i] - b[i];
		}
		if(diff[a.length-1] > 0){
			diff[a.length-1] = 1;
		}
		if(diff[a.length-1] < 0){
			diff[a.length-1] = -1;
		}
		if(diff[a.length-1] == 0){
			return null;
		}
		Instance diffInst = new DenseInstance(1.0,diff);
		return diffInst;
	}

	@Override
	public Classifier training(Instances dataset) {
		try{
			model.buildClassifier(dataset);
		}catch(Exception e){
			System.err.println("Error training PairwiseLearner LibSVM model");
			e.printStackTrace();
		}
		return model;
	}

	@Override
	public TestFeatures extract_test_features(String test_data_file,
			Map<String, Double> idfs) {
		
		Standardize filter = new Standardize();
		Instances dataset = null;
		
		/* Build attributes list */
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("body_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));
		attributes.add(new Attribute("relevance_score"));
		dataset = new Instances("train_dataset", attributes, 0);
		
		/* Add data */
		// query -> documents list
		Map<Query,List<Document>> testData = null;
		try{
			testData = Util.loadTrainData(test_data_file);
		}catch(Exception e){
			System.err.println("Error loading training data");
		}
		
		Map<String, Map<String, Integer>> index_map = new HashMap<String, Map<String,Integer>>();
		
		int index = 0;
		for (Map.Entry<Query, List<Document>> entry : testData.entrySet()){
			Query q = entry.getKey();
			Map<String,Integer> mp = new HashMap<String,Integer>();
			List<Document> documents = entry.getValue();
			for(Document d : documents){
				double[] instance = FormatDocument.createInstanceVector(d, q, idfs, null);
				Instance inst = new DenseInstance(1.0,instance);
				dataset.add(inst);
				mp.put(d.url, new Integer(index));
				index = index + 1;
			}
			index_map.put(q.query,mp);
		}
		/* Set last attribute as target */
		dataset.setClassIndex(dataset.numAttributes() - 1);
		Instances new_instances = null;
		try{
			filter.setInputFormat(dataset);
			new_instances = Filter.useFilter(dataset, filter);
		}catch(Exception e){
			System.err.println("Error filtering dataset");
		}
		TestFeatures tf = new TestFeatures();
		tf.features = new_instances;
		tf.index_map = index_map;
		return tf;
	}

	@Override
	public Map<String, List<String>> testing(TestFeatures tf,
			Classifier model) {
		
		Map<String, List<String>> rankings = new HashMap<String, List<String>>();
		Instances test_dataset = tf.features;
		Map<String, Map<String,Integer>> index_map = tf.index_map;
		for (Map.Entry<String, Map<String,Integer>> entry1 : index_map.entrySet()){
			String query = entry1.getKey();
			Map<String,Integer> docMap = entry1.getValue();
			List<Pair<String,Double>> list = new ArrayList<Pair<String,Double>>();
			for (Map.Entry<String,Integer> entry2 : docMap.entrySet()){
				double prediction = Double.MIN_VALUE;
				String url = entry2.getKey();
				Integer index = entry2.getValue();
				try{
					prediction = model.classifyInstance(test_dataset.get(index.intValue()));
					//System.err.println("prediction: " + prediction);
				}catch(Exception e){
					System.err.println("Error classifying url " + url);
				}
				Pair<String,Double> p = new Pair<String,Double>(url,new Double(prediction));
				list.add(p);
			}
			FormatDocument.sortList(list);
			rankings.put(query,FormatDocument.convertList(list));
		}
		return rankings;
	}

}
