-- ============================================================
-- CINNAMON DISEASE DETECTION DATABASE SETUP
-- ============================================================

-- Create database
CREATE DATABASE IF NOT EXISTS cinnamon_db;
USE cinnamon_db;

-- Create predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    disease VARCHAR(100) NOT NULL,
    confidence FLOAT NOT NULL,
    image_data LONGTEXT NOT NULL,
    severity VARCHAR(20) NOT NULL,
    all_probabilities JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_created_at (created_at),
    INDEX idx_disease (disease),
    INDEX idx_severity (severity)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Optional: Create user for the application
-- CREATE USER 'cinnamon_user'@'localhost' IDENTIFIED BY 'your_secure_password';
-- GRANT ALL PRIVILEGES ON cinnamon_db.* TO 'cinnamon_user'@'localhost';
-- FLUSH PRIVILEGES;