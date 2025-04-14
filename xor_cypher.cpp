#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>
#include <cstring>

using namespace std;

#define Max_ARG_LENGTH 300

// Function to zero out sensitive data from memory
void zero_out_key(char* key, size_t length){
    memset(key,0,length);
}

// Hash function to generate a simple hash (for checksum)
size_t hash_digest(const string& data){
    hash<string> hash_fun;
    return hash_fun(data);
}

bool authenticate(){
    string passkey;
    cout<<"Enter Password: ";
    cin>>passkey;
    cout<<endl;
    size_t digest = hash_digest(passkey);
    const size_t expected = digest; // TEMP: Print & use actual digest
    cout << "Password Digest: " << digest << endl;
    // Replace with actual value after you see this
    return digest == 486217970;
}

// XOR encryption/decryption
string XOR(const string& text,const char* key){
    string result = text;
    size_t keyLength =strlen(key);
    for(size_t i=0;i<text.length();i++){
        result[i] ^= key[i%keyLength];
    }
    return result;
}

// Binary-safe file reader
string read_file(const string& file_path) {
    ifstream file(file_path, ios::in|ios::binary);
    if (!file.is_open()){
        throw runtime_error("Failed to open file: "+file_path);
    }
    string contents((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
    file.close();
    return contents;
}

// Binary-safe file writer
void write_file(const string& file_path, const string& contents){
    ofstream file(file_path, ios::out|ios::binary);
    if (!file.is_open()) {
       throw runtime_error("Failed to open file: "+file_path);
}
    file.write(contents.c_str(), contents.size());
    file.close();
}

// Encrypt file
void encrypt(const string& input_path, const string& output_path, const string& key_path){
string input_content = read_file(input_path);
if (input_content.empty()){
    cerr<<"ERROR: Input file is empty.\n";
    return;
}

string key_str = read_file(key_path);
char key[key_str.size() + 1];
strcpy(key, key_str.c_str());

string output = XOR(input_content, key);

size_t checksum = hash_digest(input_content);
string final_output = to_string(checksum)+"\n"+output;

zero_out_key(key, sizeof(key));
write_file(output_path, final_output);

cout<<"Encryption successful.\n";
}

// Decrypt file
void decrypt(const string& encrypted_path, const string& key_path) {
ifstream file(encrypted_path, ios::in|ios::binary);
if (!file.is_open()) {
    cerr<<"ERROR: Cannot open encrypted file.\n";
    return;
}

string stored_checksum;
getline(file, stored_checksum);

string encrypted_data((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
file.close();

string key_str = read_file(key_path);
char key[key_str.size() + 1];
strcpy(key, key_str.c_str());

string decrypted = XOR(encrypted_data, key);
zero_out_key(key, sizeof(key));

size_t recalculated_checksum = hash_digest(decrypted);
if (to_string(recalculated_checksum) == stored_checksum) {
    cout<<"Integrity check passed. File decrypted.\n";
    write_file(encrypted_path, decrypted);  // Overwrite with plain text
} else {
    cerr<<"Integrity check failed! Possible tampering.\n";
}
}

// Main logic
int main(int argc, char* argv[]) {
if (argc != 4) {
    cout<<"Usage: "<<argv[0]<<" <INPUT_FILE> <OUTPUT_FILE> <KEY_FILE>\n";
    return 1;
}

string input_file = argv[1];
string output_file = argv[2];
string key_file = argv[3];

if (!authenticate()) {
    cout <<"AUTH FAILURE: Incorrect password.\n";
    return 1;
}

string choice;
cout<<"Choose an option:\n"
    <<"  1) Encrypt\n"
    <<"  2) Decrypt\n"
    <<"  3) Exit\n> ";
cin>>choice;

if (choice=="1") {
    encrypt(input_file, output_file, key_file);
} else if (choice=="2") {
    decrypt(output_file, key_file);  // Decrypts from output file
} else {
    cout<<"Exiting.\n";
}

return 0;
}
