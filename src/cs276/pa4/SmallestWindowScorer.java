package cs276.pa4;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A skeleton for implementing the Smallest Window scorer in Task 3.
 * Note: The class provided in the skeleton code extends BM25Scorer in Task 2. However, you don't necessarily
 * have to use Task 2. (You could also use Task 1, in which case, you'd probably like to extend CosineSimilarityScorer instead.)
 */
public class SmallestWindowScorer extends CosineSimilarityScorer {

	/////// Smallest window specific hyper-parameters ////////
	double B = 1.8;
	//Equal to the superclass when boostmod = 0
	//boostmod controls how much weighting we give to the boost
	double boostmod = 1;
	//Shortest distance to see all query terms
	int Q = 0;

	//////////////////////////////
	
	private double smallestWindow = Double.MAX_VALUE;
	
	public SmallestWindowScorer(Map<String, Double> idfs,Map<Query,Map<String, Document>> queryDict) {
		//extends BM25Scorer
		//super(idfs, queryDict);
		//extends CosineSimilarityScorer
		super(idfs);
	}

	public void handleSmallestWindow(Document d, Query q){
		//Holds the position of a last occurence of a word in the map
		Map<String,Integer> occurMap = new HashMap<String,Integer>();
		for(String term: q.words){
			occurMap.put(term.toLowerCase(), new Integer(-1));
		}
		for(String type: TFTYPES){
			double window = Double.MAX_VALUE;
			if(type.equals("url") && d.url != null){
				String[] tokens = d.url.split("\\W+");
				window = checkWindow(occurMap,tokens,false);
			}
			if(type.equals("title") && d.title != null){
				String[] tokens = d.title.split("\\s+");
				window = checkWindow(occurMap,tokens,false);
			}
			if(type.equals("body") && d.body_hits != null){
				String[] tokens = combineBody(d.body_hits);
				window = checkWindow(occurMap,tokens,true);
			}
			if(type.equals("header") && d.headers != null){
				List<String> headers = d.headers;
				for(String head: headers){
					window = Double.MAX_VALUE;
					String[] tokens = head.split("\\s+");
					window = checkWindow(occurMap,tokens,false);
				}
			}
			if(type.equals("anchor") && d.anchors != null){
				Map<String,Integer> links = d.anchors;
				for(Map.Entry<String, Integer> entry: links.entrySet()){
					window = Double.MAX_VALUE;
					String[] tokens = entry.getKey().split("\\s+");
					window = checkWindow(occurMap,tokens,false);
				}
			}
			if(window < smallestWindow){
				smallestWindow = window;
			}
		}
	}
	
	//Combine all of the terms of the body, put into a form such as
	//term1 -> [1,2,8,20]
	//term2 -> [3,5,28]
	//String[] -> [term1 1,term1 2,term2 3,term2 5,term1 8...]
	private String[] combineBody(Map<String, List<Integer>> body_hits){
		ArrayList<String> list = new ArrayList<String>();
		for(Map.Entry<String, List<Integer>> entry: body_hits.entrySet()){
			String key = entry.getKey();
			List<Integer> posList = entry.getValue();
			for(int pos: posList){
				String word = key + " " + pos;
				list.add(word);
			}
		}
		//Put the list in order of occurrence
		Collections.sort(list, new Comparator<String>() {
			@Override
			public int compare(String str1, String str2) {
				int str1Pos = Integer.parseInt(str1.split("\\s+")[1]);
				int str2Pos = Integer.parseInt(str2.split("\\s+")[1]);
				if(str1Pos < str2Pos){
					return -1;
				}else{
					return 1;
				}
			}	
		});
		String[] strList = new String[list.size()];
		strList = list.toArray(strList);
		return strList;
	}

	public double checkWindow(Map<String,Integer> terms,String[] docstr,boolean isBodyField){
		double windowSize = Double.MAX_VALUE;
		for(int index = 0; index < docstr.length; index++){
			String str = docstr[index];
			int pos = index;
			if(isBodyField){
				String[] splits = str.split("\\s+");
				str = splits[0];
				pos = Integer.parseInt(splits[1]);
			}
			if(terms.containsKey(str)){
				terms.put(str, new Integer(pos));
				double ret = isWindow(terms);
				if(ret < windowSize) windowSize = ret;
			}
		}
		wipeTerms(terms);
		return windowSize;
	}
	
	private void wipeTerms(Map<String,Integer> terms){
		for(Map.Entry<String,Integer> entry: terms.entrySet()){
			entry.setValue(new Integer(-1));
		}
	}
	
	public double isWindow(Map<String,Integer> terms){
		double window = Double.MAX_VALUE;
		int lowestPos = Integer.MAX_VALUE;
		int highestPos = Integer.MIN_VALUE;
		for(Map.Entry<String,Integer> entry: terms.entrySet()){
			int pos = entry.getValue();
			if(pos == -1) return window;
			if(pos < lowestPos) lowestPos = pos;
			if(pos > highestPos) highestPos = pos;
		}
		window = highestPos - lowestPos + 1;
		return window;
	}
	
	private double boostValue(){
		/*
		 * Goes to 1 if the window size is large
		 * Goes to B is the windows size is = Q.
		 */
		return ((B - 1)*(Q/smallestWindow)) + 1;
	}
	
	@Override
	public double getSimScore(Document d, Query q) {
		double score = super.getSimScore(d,q);
		HashSet<String> set = new HashSet<String>(q.words);
		Q = set.size();
		smallestWindow = Double.MAX_VALUE;
		handleSmallestWindow(d,q);
		double boostValue = boostValue();
		double modifier = boostmod*boostValue;
		score = score + score*modifier;
		return score;
	}

}
