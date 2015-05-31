package cs276.pa4;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

public class FormatDocument {
	
	//*********** Straight from Assignment 3 Start ***********//
	
	static String[] TFTYPES = {"url","title","body","header","anchor"};
	
	//Add word to Map
	private static void addKeyNIncrement(String key, Map<String,Double> tf, int inc){
		if(tf.containsKey(key)){
			tf.put(key, new Double(tf.get(key) + inc));
		}else{
			tf.put(key, new Double(inc));
		}
	}
	//Given a url string and a query word, tokenize on non alphaneumeric characters
	//and count the times the query word appears
	private static void urlTF(String qWord, String[] url, Map<String,Double> tf){
		for(String domain: url){
			if(qWord.equals(domain)){
				addKeyNIncrement(domain, tf, 1);
			}
		}
	}
	//Given a title string, tokenize on whitespace
	//and count the times the query word appears
	private static void titleTF(String qWord, String[] title, Map<String,Double> tf){
		if(title == null) return;
		for(String h: title){
			if(qWord.equals(h)){
				addKeyNIncrement(h, tf, 1);
			}
		}
		
	}
	//Loop through the headers list, tokenize the strings on whitespace
	//and count the times the query word appears
	private static void headerTF(String qWord, List<String> headers, Map<String,Double> tf){
		if(headers == null) return;
		for(String head: headers){
			String[] splits = head.split("\\s+");
			for(String split: splits){
				if(qWord.equals(split)){
					addKeyNIncrement(split, tf, 1);
				}
			}
		}
	}
	//Loop through the body_hits key set and if the key is equal to a query word,
	//get the size of that keys list
	private static void body_hitsTF(String qWord, Map<String,List<Integer>> body_hits, Map<String,Double> tf){
		if(body_hits == null) return;
		for (Map.Entry<String, List<Integer>> entry : body_hits.entrySet()){
			String key = entry.getKey();
			if(qWord.equals(key)){
				//tf.put(key, new Double(entry.getValue().size()));
				addKeyNIncrement(key, tf, entry.getValue().size());
			}
		}
	}
	//Loop the anchors key_set and tokenize the key. If the key is equal to the query word,
	//add that keys value.
	private static void anchorsTF(String qWord, Map<String, Integer> anchors, Map<String,Double> tf){
		if(anchors == null) return;
		for (Map.Entry<String, Integer> entry : anchors.entrySet()){
			String key = entry.getKey();
			String[] links = key.split("\\s+");
			HashSet<String> mySet = new HashSet<String>(Arrays.asList(links)); 
			if(mySet.contains(qWord)){
				addKeyNIncrement(qWord, tf, entry.getValue());
			}
		}
	}
	
	/*/
	 * Creates the various kinds of term frequencies (url, title, body, header, and anchor)
	 * You can override this if you'd like, but it's likely that your concrete classes will share this implementation.
	 */
	private static Map<String,Map<String, Double>> getDocTermFreqs(Document d, Query q) {
		// Map from tf type -> queryWord -> score
		Map<String,Map<String, Double>> tfs = new HashMap<String,Map<String, Double>>();
		Map<String,Double> url = new HashMap<String,Double>();
		Map<String,Double> title = new HashMap<String,Double>();
		Map<String,Double> body = new HashMap<String,Double>();
		Map<String,Double> header = new HashMap<String,Double>();
		Map<String,Double> anchor = new HashMap<String,Double>();
		tfs.put(TFTYPES[0], url);
		tfs.put(TFTYPES[1], title);
		tfs.put(TFTYPES[2], body);
		tfs.put(TFTYPES[3], header);
		tfs.put(TFTYPES[4], anchor);
		// Loop through query terms and increase relevant tfs. Note: you should do this to each type of term frequencies.
		HashSet<String> hs = new HashSet<String>(q.words);
		String[] urlSplits = (d.url.toLowerCase()).split("\\W+");
		String[] titleSplits = null;
		if(d.title != null) titleSplits = (d.title).split("\\s+");
		for (String queryWord: hs) {
			urlTF(queryWord,urlSplits,url);
			titleTF(queryWord,titleSplits,title);
			headerTF(queryWord,d.headers,header);
			body_hitsTF(queryWord,d.body_hits,body);
			anchorsTF(queryWord,d.anchors,anchor);
		}
		//return normalizeDoc(tfs);
		return tfs;
	}
	
	//*********** Straight from Assignment 3 End ***********//
	
	//Need to count the terms in each dimension, compute the distance and divide each term by it.
	private static Map<String, Double> normalizeDimension(Map<String, Double> termFreqs) {
		double count = 0;
		for(Map.Entry<String,Double> entry : termFreqs.entrySet()){
			count = count + entry.getValue().doubleValue();
		}
		Map<String, Double> normalized = new HashMap<String,Double>();
		for(Map.Entry<String,Double> entry : termFreqs.entrySet()){
			double value = entry.getValue().doubleValue()/count;
			normalized.put(entry.getKey(),new Double(value));
		}
		return normalized;
	}
	
	private static Map<String, Map<String, Double>> normalizeDoc(Map<String,Map<String,Double>> termFreqs){
		Map<String, Map<String,Double>> normalDoc = new HashMap<String,Map<String,Double>>();
		for(Map.Entry<String, Map<String, Double>> entry: termFreqs.entrySet()){
			Map<String,Double> raw = entry.getValue();
			Map<String,Double> normalized = normalizeDimension(raw);
			normalDoc.put(entry.getKey(), normalized);
		}
		return normalDoc;
	}
	
	private static Map<String,Double> getQueryIDFS(Query q, Map<String, Double> idfs){
		Map<String,Double> query = new HashMap<String,Double>(1);
		List<String> words = q.words;
		HashSet<String> seen = new HashSet<String>();
		for(String word : words){
			if(!seen.contains(word)){
				if(idfs.containsKey(word)){
					query.put(word,idfs.get(word));
				}else{
					query.put(word,1.0);
				}
			}
		}
		return query;
	}
	
	private static double dotProduct(Map<String,Double> a, Map<String,Double> b){
		double score = 0.0;
		for (Map.Entry<String, Double> entry : a.entrySet()){
			String key = entry.getKey();
			if(b.containsKey(key)){
				score = score + entry.getValue()*b.get(key);
			}
		}
		return score;
	}
	
	public static double[] createInstanceVector(Document d, Query q, Map<String, Double> idfs, Map<String, Map<String, Double>> trainRels){
		Map<String,Map<String, Double>> docMap = getDocTermFreqs(d,q);
		Map<String,Double> queryMap = getQueryIDFS(q,idfs);
		double[] vector = new double[TFTYPES.length + 1];
		for(int i = 0; i < TFTYPES.length; i++){
			String key = TFTYPES[i];
			double score = dotProduct(queryMap,docMap.get(key));
			vector[i] = score;
		}
		if(trainRels != null){
			Map<String,Double> urlScores = trainRels.get(q.query);
			double rel = urlScores.get(d.url);
			vector[TFTYPES.length] = rel;
		}else{
			vector[TFTYPES.length] = 0;
		}
		return vector;
	}
}
