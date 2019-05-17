#include "jsonPerceptronArchitectureParser.hpp"

bool isJsonPerceptronArchitectureWellWritten(const Document &doc) {
	// Check the document is well formed but does not check its content
	assert(doc.IsObject());
	bool result = true;
	// Check all mandatory values are member of the Json and their types is correct
	if (!doc.HasMember("numberLayers") || !doc["numberLayers"].IsInt()) {
		std::cout << "Your perceptron architecture json has no 'numberLayers' member or is poorly defined." << std::endl;
		return false;
	}
	//
	unsigned int numberOfLayers = doc["numberLayers"].GetInt();
	//
	if (!doc.HasMember("learningRate") || !doc["learningRate"].IsFloat()) {
		std::cout << "Your perceptron architecture json has no 'learningRate' member or is poorly defined." << std::endl;
		result = false;
	}
	//
	if (!doc.HasMember("inertiaRate") || !doc["inertiaRate"].IsFloat()) {
		std::cout << "Your perceptron architecture json has no 'inertiaRate' member or is poorly defined." << std::endl;
		result = false;
	}
	//
	if (!doc.HasMember("biasByLayers") || !doc["biasByLayers"].IsArray()) {
		std::cout << "Your perceptron architecture json has no 'biasByLayers' member or is poorly defined." << std::endl;
		result = false;
	} else {
		const Value& biasByLayersArray = doc["biasByLayers"];
		if (biasByLayersArray.Size() != numberOfLayers) {
			std::cout << "The array size of 'biasByLayers' member is not correct." << std::endl;
			result = false;
		}
		if (!biasByLayersArray[0].IsBool()) {
			std::cout << "Your perceptron architecture 'biasByLayers' members should be booleans." << std::endl;
			result = false;
		}
	}
	//
	if (!doc.HasMember("neuronsByLayers") || !doc["neuronsByLayers"].IsArray()) {
		std::cout << "Your perceptron architecture json has no 'neuronsByLayers' member or is poorly defined." << std::endl;
		result = false;
	} else {
		const Value& neuronsByLayersArray = doc["neuronsByLayers"];
		if (neuronsByLayersArray.Size() != numberOfLayers) {
			std::cout << "The array size of 'neuronsByLayers' member is not correct." << std::endl;
			result = false;
		}
		if (!neuronsByLayersArray[0].IsInt()) {
			std::cout << "Your perceptron architecture 'neuronsByLayers' members should be integers." << std::endl;
			result = false;
		}
	}
	//
	if (!doc.HasMember("activationFunctionByLayers") || !doc["activationFunctionByLayers"].IsArray()) {
		std::cout << "Your perceptron architecture json has no 'activationFunctionByLayers' member or is poorly defined." << std::endl;
		result = false;
	} else {
		const Value& activationFunctionByLayersArray = doc["activationFunctionByLayers"];
		if (activationFunctionByLayersArray.Size() != numberOfLayers) {
			std::cout << "The array size of 'activationFunctionByLayers' member is not correct." << std::endl;
			result = false;
		}
		if (activationFunctionByLayersArray.Size() >=2) {
			if (!activationFunctionByLayersArray[0].IsNull()) {
				std::cout << "First element of 'activationFunctionByLayers' members should be 'null'." << std::endl;
				result = false;
			}
			if (!activationFunctionByLayersArray[1].IsString()) {
				std::cout << "All elements of 'activationFunctionByLayers' members should be strings, except the first one." << std::endl;
				result = false;
			}
		} else {
			std::cout << "First element of 'activationFunctionByLayers' members should be 'null'." << std::endl;
			std::cout << "All elements of 'activationFunctionByLayers' members should be strings, except the first one." << std::endl;
			result = false;
		}
	}
	//
	if (!doc.HasMember("seed") || !doc["seed"].IsInt()) {
		std::cout << "Your perceptron architecture json has no 'seed' member or is poorly defined." << std::endl;
		result = false;
	}
	//
	if (!doc.HasMember("distribution") || !doc["distribution"].IsString()) {
		std::cout << "Your perceptron architecture json has no 'distribution' member or is poorly defined." << std::endl;
		result = false;
	}
	// 
	if (!doc.HasMember("paramOfDist1") || !doc["paramOfDist1"].IsFloat() ) {
		std::cout << "Your perceptron architecture json has no 'paramOfDist1' member or is poorly defined." << std::endl;
		result = false;
	}
	//
	if (!doc.HasMember("paramOfDist2") || !doc["paramOfDist2"].IsFloat()) {
		std::cout << "Your perceptron architecture json has no 'paramOfDist2' member or is poorly defined." << std::endl;
		result = false;
	}
	return result;
}

bool readJsonPerceptronArchitecture(
    const char * json,
	float &inertiaRate,
	float &learningRate,
	int &seed,
	std::string &dist,
	float &paramOfDist1,
	float &paramOfDist2,
    network &neuronalNetwork
) {
    // Open json file and parse it
	FILE* fp = fopen(json, "r");
	if (fp == NULL) {
		fprintf(stderr,"Impossible d'ouvrir le fichier données en lecture\n");
		exit(1);
	}
	char readBuffer[4096];
	FileReadStream is(fp, readBuffer, sizeof(readBuffer));
    Document jsonInputs;
    jsonInputs.ParseStream(is);
    // Check the file is well writen
    if (isJsonPerceptronArchitectureWellWritten(jsonInputs)) {
		inertiaRate = jsonInputs["inertiaRate"].GetFloat();
		learningRate = jsonInputs["learningRate"].GetFloat();
		neuronalNetwork.numberLayers = jsonInputs["numberLayers"].GetInt();
		const Value& biasByLayersArray = jsonInputs["biasByLayers"];
		const Value& neuronsByLayersArray = jsonInputs["neuronsByLayers"];
		const Value& activationFunctionByLayersArray = jsonInputs["activationFunctionByLayers"];
		neuronalNetwork.biasByLayers = static_cast<bool *>(malloc(neuronalNetwork.numberLayers * sizeof(bool)));
		neuronalNetwork.neuronsByLayers = static_cast<int *>(malloc(neuronalNetwork.numberLayers * sizeof(int)));
		neuronalNetwork.activationFunctionByLayers = static_cast<ActivationFunction *>(malloc(neuronalNetwork.numberLayers * sizeof(ActivationFunction)));
		neuronalNetwork.activationFunctionDerivByLayers = static_cast<ActivationDerivFunction *>(malloc(neuronalNetwork.numberLayers * sizeof(ActivationDerivFunction)));
		for (int i = 0 ; i<neuronalNetwork.numberLayers ; i++) {
			neuronalNetwork.biasByLayers[i] = biasByLayersArray[i].GetBool();
			neuronalNetwork.neuronsByLayers[i] = neuronsByLayersArray[i].GetInt();
			if (i == 0) {
				neuronalNetwork.activationFunctionByLayers[i] = NONE;
				neuronalNetwork.activationFunctionDerivByLayers[i] = NONE_DERIV;
			} else {
				activationFunctionEnumFromString(activationFunctionByLayersArray[i].GetString(), i, neuronalNetwork.activationFunctionByLayers, neuronalNetwork.activationFunctionDerivByLayers);
			}
		}
		seed = jsonInputs["seed"].GetInt();
		dist = jsonInputs["distribution"].GetString();
		paramOfDist1 = jsonInputs["paramOfDist1"].GetFloat();
		paramOfDist2 = jsonInputs["paramOfDist2"].GetFloat();
		fclose(fp);
		return true;
    }
	fclose(fp);
	return false;
}

bool writeJsonPerceptronArchitecture(
    const char * json,
	const float &inertiaRate,
	const float &learningRate,
	const int &seed,
	const std::string &dist,
	const float &paramOfDist1,
	const float &paramOfDist2,
    network &neuronalNetwork
) {
	// Create the  json 
	Document output;
	output.Parse(json);
	// Set up all values in json
	// Define the document as an object rather than an array
	output.SetObject();
	// Must pass an allocator when the object may need to allocate memory
	Document::AllocatorType& allocator = output.GetAllocator();
	//
	output.AddMember("inertiaRate", inertiaRate, allocator);
	//
	output.AddMember("learningRate", learningRate, allocator);
	//
	output.AddMember("numberLayers", neuronalNetwork.numberLayers, allocator);
	//
	Value biasByLayersArray(kArrayType);
	for (int i = 0; i < neuronalNetwork.numberLayers ; i++)
		biasByLayersArray.PushBack(neuronalNetwork.biasByLayers[i], allocator);
	output.AddMember("biasByLayers", biasByLayersArray, allocator);
	//
	Value neuronsByLayersArray(kArrayType);
	for (int i = 0; i < neuronalNetwork.numberLayers ; i++)
		neuronsByLayersArray.PushBack(neuronalNetwork.neuronsByLayers[i], allocator);
	output.AddMember("neuronsByLayers", neuronsByLayersArray, allocator);
	//
	Value activationFunctionByLayersArray(kArrayType);
	for (int i = 0 ; i < neuronalNetwork.numberLayers ; i ++) {
		if (neuronalNetwork.activationFunctionByLayers[i] != 0) {
			std::string functionName = activationFunctionStringFromEnum(neuronalNetwork.activationFunctionByLayers[i]);
			Value functionStr;
			functionStr.SetString(functionName.c_str(), (unsigned int) functionName.length(), allocator);
			activationFunctionByLayersArray.PushBack(functionStr, allocator);
		} else {
			Value a(1), b(2);
			b = a;
			activationFunctionByLayersArray.PushBack(a, allocator);
		}
	}
	output.AddMember("activationFunctionByLayers", activationFunctionByLayersArray, allocator);
	//
	output.AddMember("seed", seed, allocator);
	//
	Value distStr;
	distStr.SetString(dist.c_str(), (unsigned int) dist.length(), allocator);
	output.AddMember("distribution", distStr, allocator);
	//
	output.AddMember("paramOfDist1", paramOfDist1, allocator);
	//
	output.AddMember("paramOfDist2", paramOfDist2, allocator);
	// Write the json file on disk
	FILE* fp = fopen(json, "w");
	if (fp == NULL) {
		fprintf(stderr,"Impossible d'ouvrir le fichier données en écriture\n");
		exit(1);
	}
	char writeBuffer[4096];
	FileWriteStream os(fp, writeBuffer, sizeof(writeBuffer));
	Writer<FileWriteStream> writer(os);
	bool res = output.Accept(writer);
	fclose(fp);
	return res;
}