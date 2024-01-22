DROP TABLE IF EXISTS patient;
CREATE TABLE patient
(
    uniquepid VARCHAR(10) NOT NULL,
    patienthealthsystemstayid INT NOT NULL,
    patientunitstayid INT NOT NULL PRIMARY KEY,
    gender VARCHAR(25) NOT NULL,
    age VARCHAR(10) NOT NULL,
    ethnicity VARCHAR(50),
    hospitalid INT NOT NULL,
    wardid INT NOT NULL,
    admissionheight NUMERIC(10,2),
    admissionweight NUMERIC(10,2),
    dischargeweight NUMERIC(10,2),
    hospitaladmittime TIMESTAMP(0) NOT NULL,
    hospitaladmitsource VARCHAR(30) NOT NULL,
    unitadmittime TIMESTAMP(0) NOT NULL,
    unitdischargetime TIMESTAMP(0),
    hospitaldischargetime TIMESTAMP(0),
    hospitaldischargestatus VARCHAR(10)
) ;

DROP TABLE IF EXISTS diagnosis;
CREATE TABLE diagnosis
(
    diagnosisid INT NOT NULL PRIMARY KEY,
    patientunitstayid INT NOT NULL,
    diagnosisname VARCHAR(200) NOT NULL,
    diagnosistime TIMESTAMP(0) NOT NULL,
    icd9code VARCHAR(100),
    FOREIGN KEY(patientunitstayid) REFERENCES patient(patientunitstayid)
) ;

DROP TABLE IF EXISTS treatment;
CREATE TABLE treatment
(
    treatmentid INT NOT NULL PRIMARY KEY,
    patientunitstayid INT NOT NULL,
    treatmentname VARCHAR(200) NOT NULL,
    treatmenttime TIMESTAMP(0) NOT NULL,
    FOREIGN KEY(patientunitstayid) REFERENCES patient(patientunitstayid)
) ;

DROP TABLE IF EXISTS lab;
CREATE TABLE lab
(
    labid INT NOT NULL PRIMARY KEY,
    patientunitstayid INT NOT NULL,
    labname VARCHAR(256) NOT NULL,
    labresult NUMERIC(11,4) NOT NULL,
    labresulttime TIMESTAMP(0) NOT NULL,
    FOREIGN KEY(patientunitstayid) REFERENCES patient(patientunitstayid)
) ;

DROP TABLE IF EXISTS medication;
CREATE TABLE medication
(
    medicationid INT NOT NULL PRIMARY KEY,
    patientunitstayid INT NOT NULL,
    drugname VARCHAR(220) NOT NULL,
    dosage VARCHAR(60) NOT NULL,
    routeadmin VARCHAR(120) NOT NULL,
    drugstarttime TIMESTAMP(0),
    drugstoptime TIMESTAMP(0),
    FOREIGN KEY(patientunitstayid) REFERENCES patient(patientunitstayid)
) ;

DROP TABLE IF EXISTS cost;
CREATE TABLE cost
(
    costid INT NOT NULL PRIMARY KEY,
    uniquepid VARCHAR(10) NOT NULL,
    patienthealthsystemstayid INT NOT NULL,
    eventtype VARCHAR(20) NOT NULL,
    eventid INT NOT NULL,
    chargetime TIMESTAMP(0) NOT NULL,
    cost DOUBLE PRECISION NOT NULL,
    FOREIGN KEY(patienthealthsystemstayid) REFERENCES patient(patienthealthsystemstayid)
) ;

DROP TABLE IF EXISTS allergy;
CREATE TABLE allergy
(
    allergyid INT NOT NULL PRIMARY KEY,
    patientunitstayid INT NOT NULL,
    drugname VARCHAR(255),
    allergyname VARCHAR(255) NOT NULL,
    allergytime TIMESTAMP(0) NOT NULL,
    FOREIGN KEY(patientunitstayid) REFERENCES patient(patientunitstayid)
) ;

DROP TABLE IF EXISTS intakeoutput;
CREATE TABLE intakeoutput
(
    intakeoutputid INT NOT NULL PRIMARY KEY,
    patientunitstayid INT NOT NULL,
    cellpath VARCHAR(500) NOT NULL,
    celllabel VARCHAR(255) NOT NULL,
    cellvaluenumeric NUMERIC(12,4) NOT NULL,
    intakeoutputtime TIMESTAMP(0) NOT NULL,
    FOREIGN KEY(patientunitstayid) REFERENCES patient(patientunitstayid)
) ;

DROP TABLE IF EXISTS microlab;
CREATE TABLE microlab
(
    microlabid INT NOT NULL PRIMARY KEY,
    patientunitstayid INT NOT NULL,
    culturesite VARCHAR(255) NOT NULL,
    organism VARCHAR(255) NOT NULL,
    culturetakentime TIMESTAMP(0) NOT NULL,
    FOREIGN KEY(patientunitstayid) REFERENCES patient(patientunitstayid)
) ;

DROP TABLE IF EXISTS vitalperiodic;
CREATE TABLE vitalperiodic
(
    vitalperiodicid BIGINT NOT NULL PRIMARY KEY,
    patientunitstayid INT NOT NULL,
    temperature NUMERIC(11,4),
    sao2 INT,
    heartrate INT,
    respiration INT,
    systemicsystolic INT,
    systemicdiastolic INT,
    systemicmean INT,
    observationtime TIMESTAMP(0) NOT NULL,
    FOREIGN KEY(patientunitstayid) REFERENCES patient(patientunitstayid)
) ;