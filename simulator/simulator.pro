TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
    serial_port.cpp \
    main.cpp

include(deployment.pri)
qtcAddDeployment()

LIBS += -lzmq

HEADERS += \
    serial_port.h

