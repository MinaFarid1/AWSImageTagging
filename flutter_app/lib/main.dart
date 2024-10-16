import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:fluttertoast/fluttertoast.dart';
import 'dart:convert';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AI Image Inference',
      theme: ThemeData(
        brightness: Brightness.light,
        primaryColor: Colors.deepPurple,
        scaffoldBackgroundColor: Colors.grey[100],
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: ImageInferencePage(),
    );
  }
}

class ImageInferencePage extends StatefulWidget {
  @override
  _ImageInferencePageState createState() => _ImageInferencePageState();
}

class _ImageInferencePageState extends State<ImageInferencePage> {
  File? _image;
  Map<String, double>? _probabilities;
  bool _isLoading = false;

  final ImagePicker _picker = ImagePicker();
  final String apiUrl =
      'api_url'; // Replace with your API Gateway URL

  Future<void> _pickImage(ImageSource source) async {
    final pickedFile = await _picker.pickImage(source: source);
    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
      });
      _sendImageToApi(_image!);
    }
  }

  Future<void> _sendImageToApi(File image) async {
    setState(() {
      _isLoading = true; // Start loading
      _probabilities = null; // Clear previous results
    });

    try {
      final bytes = await image.readAsBytes();
      final response = await http.post(
        Uri.parse(apiUrl),
        headers: {
          'Content-Type': 'application/x-image',
          'x-api-key': 'api_key', // Replace with your API key
        },
        body: bytes,
      );

      if (response.statusCode == 200) {
        setState(() {
          _probabilities =
              Map<String, double>.from(json.decode(response.body)['probabilities']);
        });
      } else {
        _showToast('Error: ${response.statusCode}');
      }
    } catch (e) {
      _showToast('Error sending image: $e');
    } finally {
      setState(() {
        _isLoading = false; // Stop loading
      });
    }
  }

  void _showToast(String message) {
    Fluttertoast.showToast(
      msg: message,
      toastLength: Toast.LENGTH_SHORT,
      gravity: ToastGravity.BOTTOM,
      backgroundColor: Colors.black87,
      textColor: Colors.white,
      fontSize: 16.0,
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('AI Image Inference'),
        centerTitle: true,
        elevation: 5,
        backgroundColor: Colors.deepPurple,
        leading: Builder(
          builder: (BuildContext context) {
            return IconButton(
              icon: Icon(Icons.menu, color: Colors.white),
              onPressed: () {
                Scaffold.of(context).openDrawer();
              },
            );
          },
        ),
      ),
      drawer: _buildDrawer(),
      body: SingleChildScrollView( // Added SingleChildScrollView for scrolling
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: <Widget>[
                _image == null
                    ? Icon(
                        Icons.image_not_supported,
                        size: 150,
                        color: Colors.grey[400],
                      )
                    : ClipRRect(
                        borderRadius: BorderRadius.circular(12),
                        child: Image.file(
                          _image!,
                          height: 250,
                          width: 250,
                          fit: BoxFit.cover,
                        ),
                      ),
                SizedBox(height: 30),
                if (_isLoading)
                  CircularProgressIndicator(
                    valueColor:
                        AlwaysStoppedAnimation<Color>(Colors.deepPurple),
                  )
                else if (_probabilities != null)
                  _buildResultsCard()
                else
                  Text(
                    'Tap a button below to select an image.',
                    style: TextStyle(fontSize: 16, color: Colors.grey[700]),
                    textAlign: TextAlign.center,
                  ),
                SizedBox(height: 30),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceAround,
                  children: <Widget>[
                    _buildImageButton(
                      icon: Icons.camera_alt,
                      text: 'Camera',
                      color: Colors.deepPurple,
                      onPressed: () {
                        _pickImage(ImageSource.camera);
                        _showToast('Opening Camera...');
                      },
                    ),
                    _buildImageButton(
                      icon: Icons.photo_library,
                      text: 'Gallery',
                      color: Colors.deepPurple,
                      onPressed: () {
                        _pickImage(ImageSource.gallery);
                        _showToast('Opening Gallery...');
                      },
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildImageButton({
    required IconData icon,
    required String text,
    required Color color,
    required VoidCallback onPressed,
  }) {
    return ElevatedButton.icon(
      onPressed: onPressed,
      icon: Icon(icon, color: Colors.white),
      label: Text(text, style: TextStyle(color: Colors.white)),
      style: ElevatedButton.styleFrom(
        backgroundColor: color,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(30.0),
        ),
        padding: EdgeInsets.symmetric(horizontal: 20, vertical: 12),
        elevation: 5,
      ),
    );
  }

  Widget _buildResultsCard() {
    return Card(
      elevation: 5,
      margin: EdgeInsets.symmetric(vertical: 20, horizontal: 16),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Inference Results',
              style: TextStyle(
                  fontSize: 20, fontWeight: FontWeight.bold, color: Colors.deepPurple),
            ),
            Divider(),
            for (var entry in _probabilities!.entries)
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 5),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(
                      entry.key,
                      style: TextStyle(fontSize: 16, color: Colors.grey[800]),
                    ),
                    Text(
                      '${(entry.value * 100).toStringAsFixed(2)}%',
                      style: TextStyle(fontSize: 16, color: Colors.deepPurple),
                    ),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildDrawer() {
    return Drawer(
      child: ListView(
        padding: EdgeInsets.zero,
        children: <Widget>[
          DrawerHeader(
            decoration: BoxDecoration(
              color: Colors.deepPurple,
            ),
            child: Text(
              'Team Members',
              style: TextStyle(
                color: Colors.white,
                fontSize: 24,
              ),
            ),
          ),
          _buildDrawerItem(text: 'Mina Farid', role: 'Deployment and MLOps'),
          _buildDrawerItem(text: 'Ahmed Waheed', role: 'Data Engineer'),
          _buildDrawerItem(text: 'Menna Osama', role: 'Model Development'),
          _buildDrawerItem(text: 'Norhan El-Sayed', role: 'Model Development'),
          _buildDrawerItem(text: 'Bedour Fouad', role: 'Data Engineer'),
        ],
      ),
    );
  }

  Widget _buildDrawerItem({required String text, required String role}) {
    return ListTile(
      title: Text(text),
      onTap: () {
        _showToast('$text - $role');
      },
    );
  }
}
