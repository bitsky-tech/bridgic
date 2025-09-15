
[Bridgic Event Handling Mechanism]

This module contains the base classes and types for the Bridgic event handling mechanism.

The event handling mechanism enables (quasi) real-time communication between workers inside the Automa and the application layer outside the Automa. Events can be sent from a worker to the application layer, and feedback can be sent in the opposite direction, from the application layer back to the worker, if needed.

A worker can use the `post_event()` method to send an event to the application layer. To handle these events, the application layer can register event handlers by calling the Automa's `register_event_handler()` method. Since an Automa can be nested inside another Automa, only the event handlers registered on the top-level Automa will be invoked to handle events.

Note that there are two types of events: those that require feedback from the application layer, and those that do not. An example of the former is `ProgressEvent`, which is used to send progress information from a worker to the application layer. An example of the latter is an event that asks the user for additional input.

If an event requires feedback, the application layer can provide feedback to the worker by invoking the `send` method of the `FeedbackSender` object. The worker will then `await` the future object returned by the `post_event` method to receive the feedback.

The Bridgic event handling mechanism requires the Automa process to remain alive in memory for the entire duration of the event and feedback interaction. If the interaction period is too long and the Automa process needs to be shut down in the middle, the Bridgic human interaction mechanism should be used instead.


# Documentation for `Event Handing`



::: bridgic.core.automa.interaction.Event
    handler: python
